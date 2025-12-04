import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, GeminiLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryBase
import re
import logging
import os
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from collections import Counter
import time
logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

# Helper function that writes text to a file
def append_to_file(text: str, filename: str = "log.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text)

# Calculates voting over ranking lists and returns final list
def borda_vote(rankings):
    scores = {}
    n = len(rankings[0])
    for ranking in rankings:
        for i, item in enumerate(ranking):
            points = (n - i) #* weight # First = n points, last = 1 point
            scores[item] = scores.get(item, 0) + points
    return sorted(scores, key=lambda x: scores[x], reverse=True)

class UserReasoning(ReasoningBase):
    """
    Module to reason and summarize about each user
    """

    def __init__(self, profile_type_prompt, memory, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)

    def __call__(self, user_id: str, user_data: str, user_reviews: str, num_reviews: int, task_category: str, token_budget: int):
        """Obtain user summary from LLM"""

        prompt = f'''
You are a real user on Yelp. Your user data and review history are as follows:
{user_data}
{user_reviews}
Summarize your preferences, tastes, interests, etc. using this information into a structured summary that will help you rank candidate businesses.
Prioritize pertinent information, especially information that is significant to the task category of {task_category}.
Focus on things that would help you rank items in this category and figure out what would be your number 1 ranking.
For example, how important customer service is to you, or how sensitive you are to pricing. 
Additionally, try to use the information from the reviews themselves to try and figure out what your interests or background are.
Your output should be a list of qualities/traits about yourself and should be less than 300 tokens.
Your output should always start with the user_id and a colon.
It should be followed by a line that says "This summary was generated from N reviews", where N is {num_reviews}, the number of reviews in the history.
For example, if the task_category was Dining and the number of reviews was 10
user_id:
This summary was generated from 10 reviews
- Very sensitive to price
- Enjoys Chinese food, especially Chow Mein
- Can tolerate poor service if food is delicious
- Has children who are picky eaters, so values restaurants with kids menus
- etc.

'''

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=token_budget
        )

        try:
            pattern = re.escape(user_id)
            match = re.search(pattern, reasoning_result)
            if match:
                self.memory(user_id, reasoning_result)
            else:
                raise Exception()
            time.sleep(1)
            return reasoning_result
        except:
            print("Bad LLM output")
            # Retry with larger token budget on failure
            time.sleep(1)
            return self.__call__(user_id, user_data, user_reviews, num_reviews, task_category, token_budget * 2)
    
class ItemReasoning(ReasoningBase):
    """
    Module to reason and summarize about each item
    """

    def __init__(self, profile_type_prompt, memory, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)

    def __call__(self, item_id: str, item_info: str, item_reviews: str, task_category: str, token_budget: int):
        """Obtain item summary from LLM"""

        prompt = f'''
You are a helpful assistant that summarizes details and sentiments given businesses and their Yelp reviews. The category is: {task_category}.
Here is your item data: {item_info}.
Your task is to take the data and the following reviews and output a summary of the business.
Include details on what type of business it is and general sentiment of customers.
This summary will be used in the future to help recommend businesses to users, so orient your summary around things like quality of service, cleanliness, customer base, etc.
Your output should be a paragraph with 300 tokens or less and should always begin with a line of the item_id and a colon.
The reviews are as follows: {item_reviews}

'''

        messages = [{"role": "user", "content": prompt}] 
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=token_budget
        )

        try:
            pattern = re.escape(item_id)
            match = re.search(pattern, reasoning_result)
            if match:
                self.memory(item_id, reasoning_result)
            else:
                raise Exception()
            time.sleep(1)
            return reasoning_result
        except:
            print("Bad LLM output")
            # Retry with larger token budget on failure
            time.sleep(1)
            return self.__call__(item_id, item_info, item_reviews, task_category, token_budget * 2)

class TaskReasoning(ReasoningBase):
    """Module to reason about task"""

    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, user_id: str, user_summary: str, user_categories: str, item_list: str, item_data: str, task_category: str, token_budget: int):
        """Obtain ranking from LLM"""

        prompt = f'''
You are a real user on Yelp. Your user id and a summary of your user profile and your review history are as follows: {user_summary}
Additionally, these are the counts of the categories from the businesses you have reviewed: {user_categories}
Now you will be given 20 candidate businesses.
Your goal is to return a ranked list of your recommendation of these items based on your profile.
In the ranking, make sure that the better matches are at the front of the list.
Information of the 20 candidate items is as follows: {item_data}.
In your process, prioritize the rating and sentiment of the candidates higher than the user's preferences.
Your final output should be exactly 1 ranked item list of candidates, all on the same line.
That line MUST be a valid Python list literal containing ALL and ONLY the candidate item IDs, in ranked order.
DO NOT introduce any other item ids in the ranking list that are not candidates!
DO NOT provide your reasoning or analysis.
The correct output format:
['id1', 'id2', 'id3', ..., 'id20']

'''

        # Get n results from prompt
        messages = [{"role": "user", "content": prompt}]
        reasoning_results = self.llm(
            messages=messages,
            temperature=0.7,
            max_tokens=token_budget,
            n=3
        )
        
        try:
            ranking_list = []
            # Try to convert all outputs to lists
            for reasoning_result in reasoning_results:
                try:
                    # Get ranking list
                    match = re.search(r"\[.*?\]", reasoning_result, re.DOTALL)
                    if match:
                        result = match.group()
                        ranking = eval(result)
                        ranking_list.append(ranking)
                except:
                    continue
            # Only retry __call__ if all outputs are invalid
            if not ranking_list:
                raise Exception()
            # Do voting between all output lists
            final_ranking = borda_vote(ranking_list)
            print('Processed Output:', final_ranking)
            time.sleep(1)
            return final_ranking
        except:
            print('format error')
            # Retry with larger token budget on failure
            time.sleep(1)
            return self.__call__(user_id, user_summary, user_categories, item_list, item_data, task_category, token_budget * 2)
    
class ModuleMemory(MemoryBase):
    """Module to store information from LLM"""

    def __init__(self, memory_type: str, llm) -> None:
        self.llm = llm
        self.embedding = self.llm.get_embedding_model()
        db_path = os.path.join('./db', memory_type)
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
        )

    def __call__(self, item_id: str, item: str = None):
        if not item_id:
            return None
        if not item:
            return self.retriveMemory(item_id)
        else:
            self.addMemory(item_id, item)

    def retriveMemory(self, item_id: str):
        result = self.scenario_memory.get(ids=[item_id])
        if not result["ids"]:
            return None
        else:
            return result["documents"][0]

    def addMemory(self, item_id: str, item: str):
        memory_doc = Document(
            page_content=item,
            metadata={
                "id": item_id
            },
            id=item_id
        )
        self.scenario_memory.add_documents([memory_doc])
    
class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of RecommendationAgent
    """

    llm = GeminiLLM(api_key="")
    user_memory = ModuleMemory(memory_type="user", llm=llm)
    item_memory = ModuleMemory(memory_type="item", llm=llm)

    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.user_reasoning = UserReasoning(profile_type_prompt='', memory=MyRecommendationAgent.user_memory, llm=self.llm)
        self.item_reasoning = ItemReasoning(profile_type_prompt='', memory=MyRecommendationAgent.item_memory, llm=self.llm)
        self.task_reasoning = TaskReasoning(profile_type_prompt='', llm=self.llm)

    def parseUserData(self, user_id: str):
        user_data = self.interaction_tool.get_user(user_id=user_id)
        relevant_data = {}
        relevant_fields = ['user_id', 'name', 'review_count', 'average_stars']
        relevant_data = {field: user_data[field] for field in relevant_fields if field in user_data}
        return str(relevant_data)
    
    # Gives more context to each review in review history
    def parseReviewHistory(self, user_id: str):
        review_history = self.interaction_tool.get_reviews(user_id=user_id)
        detailed_reviews = []
        relevant_review_fields = ['stars', 'text', 'date']
        relevant_item_fields = ['name', 'categories']
        # Include item info into review history
        for review in review_history:
            item = self.interaction_tool.get_item(item_id=review['item_id'])
            relevant_review_data = {}
            for field in relevant_item_fields:
                if field in item:
                    relevant_review_data[field] = item[field]
            for field in relevant_review_fields:
                if field in review:
                    relevant_review_data[field] = review[field]
            detailed_reviews.append(relevant_review_data)
        return str(detailed_reviews), len(detailed_reviews)
    
    def parseItemData(self, item_id: str):
        item_data = self.interaction_tool.get_item(item_id=item_id)
        relevant_fields = ['item_id', 'name','stars','review_count','attributes', 'categories']
        relevant_data = {field: item_data[field] for field in relevant_fields if field in item_data}
        return relevant_data
    
    def parseItemReviews(self, item_id: str):
        reviews = self.interaction_tool.get_reviews(item_id=item_id)[:25]
        parsed_reviews = []
        relevant_review_fields = ['stars', 'text', 'date']
        for review in reviews:
            relevant_review_data = {field: review[field] for field in relevant_review_fields if field in review}
            parsed_reviews.append(relevant_review_data)
        return str(parsed_reviews)
    
    # Get counts of each category that appears in user's review history
    def getCategoryCounts(self, user_id):
        category_counter = Counter()
        reviews = self.interaction_tool.get_reviews(user_id=user_id)
        for review in reviews:
            item_id = review.get("item_id")
            if not item_id:
                continue
            item = self.interaction_tool.get_item(item_id=item_id)
            if not item:
                continue
            categories = item.get("categories")
            if not categories:
                continue
            categories = [c.strip().lower() for c in categories.split(",")]
            for c in categories:
                if c:
                    category_counter[c] += 1
        return dict(category_counter)

    def workflow(self, token_budget: int = 20000):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """

        plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find item information'},
         {'description': 'Next, I need to find review information'}
         ]

        user_id = ''
        user = ''
        item_list = []
        history_review = ''
        num_reviews = 0
        candidate_list = self.task['candidate_list']
        for sub_task in plan:
            if 'user' in sub_task['description']:
                user_id = self.task['user_id']
                user = self.parseUserData(user_id)
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])
            elif 'item' in sub_task['description']:
                for n_bus in range(len(candidate_list)):
                    item_list.append(self.parseItemData(candidate_list[n_bus]))
            elif 'review' in sub_task['description']:
                history_review, num_reviews = self.parseReviewHistory(user_id)
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass

        # Check if user summary already in memory
        task_category = self.task['candidate_category']
        result = self.user_memory(user_id)
        if result:
            user_summary = result
        else:
            user_summary = self.user_reasoning(user_id, user, history_review, num_reviews, task_category, token_budget)
        user_categories = self.getCategoryCounts(user_id)

        # Check if item summaries already in memory
        item_summaries = []
        for i, item_id in enumerate(candidate_list):
            result = self.item_memory(item_id)
            if result:
                item_summaries.append(result)
            else:                
                reviews = self.parseItemReviews(item_id)
                item_summaries.append(self.item_reasoning(item_id, item_list[i], reviews, task_category, token_budget))

        # Append item summary to item data dict
        for i, item in enumerate(item_list):
            item['summary'] = item_summaries[i]
        
        result = self.task_reasoning(
            user_id, 
            user_summary, 
            str(user_categories),
            str(candidate_list),
            str(item_list),
            task_category,
            token_budget
        )

        return result

if __name__ == "__main__":
    task_set = "yelp"

    # Initialize Simulator
    simulator = Simulator(data_dir="../data/yelp_processed", device="auto", cache=True)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"./track2/{task_set}/tasks", groundtruth_dir=f"./track2/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    simulator.set_llm(MyRecommendationAgent.llm)

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10) 

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
