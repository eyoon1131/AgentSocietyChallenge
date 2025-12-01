import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, OpenAILLM, GeminiLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import re
import logging
import time
import os 
logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

CACHE_PATH = "./item_summary_cache.json"

def load_item_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_item_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=4)
class ItemSummaryReasoning(ReasoningBase):
    """
    Summarize the review history for a single item into a short, informative
    natural-language summary that the ranking LLM can consume.
    """
    def __init__(self, llm, max_review_tokens: int = 2048, max_summary_tokens: int = 256):
        super().__init__(profile_type_prompt='', memory=None, llm=llm)
        self.max_review_tokens = max_review_tokens
        self.max_summary_tokens = max_summary_tokens

    def _truncate_reviews(self, reviews_text: str) -> str:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(reviews_text)
        if len(tokens) <= self.max_review_tokens:
            return reviews_text
        return encoding.decode(tokens[:self.max_review_tokens])

    def __call__(self, item: dict, reviews_text: str) -> str:
        """
        Args:
            item: raw item dict from interaction_tool.get_item(...)
            reviews_text: raw text of reviews for this item (possibly many)
        Returns:
            A short summary string.
        """
        if not isinstance(reviews_text, str):
            reviews_text = str(reviews_text)

        reviews_text = self._truncate_reviews(reviews_text)

        name = item.get('name', '')
        stars = item.get('stars', None)
        review_count = item.get('review_count', None)
        attrs = item.get('attributes', {})

        prompt = f"""
You are summarizing Yelp reviews for a single restaurant.

Basic info:
- Name: {name}
- Stars: {stars}
- Review count: {review_count}
- Key attributes (raw): {attrs}

Below are raw reviews for this business (they may be truncated):

{reviews_text}

Write a concise, factual summary of what people think about this restaurant.
Focus on:
- Overall sentiment and how good it is
- What people like (e.g., food, service, atmosphere, value)
- Any consistent complaints
- Price level / vibe if clear

Constraints:
- 3–5 sentences.
- Do NOT mention that the reviews are truncated or that you are an AI.
- Do NOT repeat the raw reviews verbatim.
"""

        messages = [{"role": "user", "content": prompt.strip()}]

        summary = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
        )

        if summary is None:
            return ""

        return str(summary).strip()

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

Task: {task_description}
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}
'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=16384,
        )
        
        return reasoning_result
import textwrap
class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent
    """
    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.planning = RecPlanning(llm=self.llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)
        self.item_summarizer = ItemSummaryReasoning(llm=self.llm)  # NEW
        # Load cache once per run
        self.item_cache = load_item_cache()
    def workflow(self,cache_lock):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        # plan = self.planning(task_type='Recommendation Task',
        #                      task_description="Please make a plan to query user information, you can choose to query user, item, and review information",
        #                      feedback='',
        #                      few_shot='')
        # print(f"The plan is :{plan}")
        plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find item information'},
         {'description': 'Next, I need to find review information'}
         ]

        user = ''
        item_list = []
        history_review = ''
        for sub_task in plan:
            
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])
            elif 'item' in sub_task['description']:
                # tmp = 0
                for cand_item_id in self.task['candidate_list']:
                    # first check cache
                    if cand_item_id in self.item_cache:
                        item_list.append(self.item_cache[cand_item_id])
                        continue
                    # tmp += 1
                    # 1) Get raw item info
                    item = self.interaction_tool.get_item(item_id=cand_item_id)
                    # 2) Get reviews for this item (assuming API signature uses item_id)
                    # try:
                    #     raw_item_reviews = self.interaction_tool.get_reviews(item_id=cand_item_id)
                    # except TypeError:
                    #     # If your API is different, adjust this, e.g. get_reviews(business_id=...)
                    #     raw_item_reviews = ""

                    # if not isinstance(raw_item_reviews, str):
                    #     # raw_item_reviews = str(raw_item_reviews)
                    #     item_review_text = [x['text'] for x in raw_item_reviews]
                    #     item_review_text = str(item_review_text)
                    
                    # # 3) Summarize the reviews using the new ReasoningBase module
                    # item_summary = self.item_summarizer(item=item, reviews_text=raw_item_reviews)
                    # # 4) Build compact item representation for the ranking LLM
                    item_entry = {
                        "item_id": item.get("item_id", cand_item_id),
                        "name": item.get("name", ""),
                        "stars": item.get("stars", None),
                        "review_count": item.get("review_count", None),
                        "summary": item.get("categories",None),
                    }
                    # save to cache and JSON file
                #     with cache_lock:
                #         self.item_cache[cand_item_id] = item_entry
                    item_list.append(item_entry)
                # if tmp != 0:
                #     with cache_lock:
                #         time.sleep(4)
                #         save_item_cache(self.item_cache)
                #         time.sleep(4)
                def format_candidate_items(items):
                    """
                    Convert list of candidate item dicts into a compact, LLM-friendly text block.

                    Expected keys per item:
                        item_id, name, stars, review_count, summary (using your item_entry mapping)

                    Returns:
                        A multi-line string like:

                        (1) Tio Pepe Restaurant & Bar  ⭐ 4.5 (88 reviews)
                            item_id: abc123
                            Summary: Portuguese cuisine, seafood, friendly atmosphere...
                    """

                    lines = []
                    for idx, it in enumerate(items, start=1):
                        name         = it.get("name", "Unknown Name")
                        item_id      = it.get("item_id", "")
                        stars        = it.get("stars", "N/A")
                        review_count = it.get("review_count", "N/A")
                        summary      = it.get("summary", "")

                        # Flatten summary/category text for readability
                        if isinstance(summary, list):
                            summary = ", ".join(summary)
                        summary = str(summary).replace("\n", " ").strip()

                        entry = (
                            f"({idx}) {name}  ⭐ {stars} ({review_count} reviews)\n"
                            f"     id: {item_id}\n"
                            f"     Summary: {summary}\n"
                        )
                        lines.append(entry)

                    return "\n".join(lines)
                item_list = format_candidate_items(item_list)


                # print(item)
            elif 'review' in sub_task['description']:
                history_review = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                def format_user_review_history(reviews, max_reviews=None):
                    """
                    reviews: list of dicts with keys like 'stars', 'useful', 'funny', 'cool', 'date', 'text'
                    max_reviews: optionally limit to first N reviews (after sorting) to save tokens
                    """
                    # (Optional) sort by usefulness + recency, most informative first
                    sorted_reviews = sorted(
                        reviews,
                        key=lambda r: (
                            r.get("useful", 0) + r.get("cool", 0) + r.get("funny", 0),
                            r.get("date", ""),
                        ),
                        reverse=True,
                    )

                    if max_reviews is not None:
                        sorted_reviews = sorted_reviews[:max_reviews]

                    lines = []
                    for i, r in enumerate(sorted_reviews, start=1):
                        stars = r.get("stars", "N/A")
                        useful = r.get("useful", 0)
                        funny  = r.get("funny", 0)
                        cool   = r.get("cool", 0)
                        date   = str(r.get("date", ""))[:10]
                        text   = r.get("text", "").strip().replace("\n", " ")

                        entry = (
                            f"- Review {i}:\n"
                            f"  stars: {stars} | useful: {useful}, funny: {funny}, cool: {cool}\n"
                            f"  date: {date}\n"
                            f"  text: {text}\n"
                        )
                        lines.append(entry)

                    return "\n".join(lines)
                history_review= format_user_review_history(history_review,max_reviews=30)
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass
        task_description = textwrap.dedent(f"""
            You are simulating a specific Yelp user.
            The review history below is the ONLY signal of this user's personal taste.

            First, internally infer what this user likes and dislikes from their review history
            (e.g., food, service, atmosphere, price, vibe). DO NOT output this reasoning.

            ========================
            USER REVIEW HISTORY
            ========================
            {history_review}

            ========================
            CANDIDATE ITEMS
            ========================

            You will now rank 20 candidate businesses by how much THIS USER would personally like them.

            Candidate item IDs (for reference):
            {self.task['candidate_list']!r}

            Information for each candidate item:
            {item_list}

            ========================
            YOUR TASK
            ========================

            Using ONLY:
            - the user's review history above, and
            - the candidate item metadata (name, stars, review_count, summary),

            rank how much this user would personally like each candidate business.

            Important:
            - Do NOT simply sort by stars or review_count.
            - Focus on how well each business matches this specific user's tastes and tone.

            ========================
            STRICT OUTPUT RULES
            ========================

            1. Output ONLY a Python list literal of item IDs.
            2. Include ALL and ONLY the IDs from the Candidate item ID list above.
            3. Do NOT explain your reasoning.
            4. Do NOT include any extra words, comments, or formatting before or after the list.

            Correct output EXAMPLE format (example order only, IDs must be from the list above):

            ['K-S_YhtoGffXF9f3azGY2A', '2hvDcM769GC5t6zrkn4A3w', 'Rzx2E5XgTeGU7FEbUZ-bGg', ..., 'Cp5vp6LVGudRE6WXwVwEaA']

            Now output your ranked list of item IDs, from MOST preferred to LEAST preferred:
        """).strip()
        print("***task: ", task_description, "--- \n\n")
        print("tokens: ",num_tokens_from_string(task_description))
        # print(task_description)
        for i in range(3):
            result = self.reasoning(task_description)
            if result:
                break
            else:
                print(f'attempt {i} failed trying again')
                print("tokens: ",num_tokens_from_string(task_description))
        
        try:
            print('Meta Output:',result)
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
                print("failed prompt: ", task_description)
            print('Processed Output:',eval(result))
            # time.sleep(4)
            return eval(result)
        except:
            print('format error')
            print("failed prompt: ", task_description)
            return ['']


if __name__ == "__main__":
    task_set = "yelp" # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(data_dir="../data/yelp_processed", device="auto", cache=True)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"./track2/{task_set}/tasks", groundtruth_dir=f"./track2/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    simulator.set_llm(GeminiLLM(api_key="AIzaSyBvIkLwUF9fHFmzd7vJVhdkL3CSof9Osy4"))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
