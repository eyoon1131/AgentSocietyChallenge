import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, OpenAILLM, GeminiLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryBase
import re
import logging
import time
import os
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import shutil
logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

def append_to_file(text: str, filename: str):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text)

# Calculates voting over ranking lists and returns final list
def weighted_borda_vote(rankings, confidences):
    scores = {}
    n = len(rankings[0])
    for ranking, weight in zip(rankings, confidences):
        for i, item in enumerate(ranking):
            points = (n - i) * weight # First = n points, last = 1 point, weighed by confidence
            scores[item] = scores.get(item, 0) + points
    return sorted(scores, key=lambda x: scores[x], reverse=True)

class UserReasoning(ReasoningBase):
    """
    Module to reason and summarize about each user
    """

    def __init__(self, profile_type_prompt, memory, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)

    def __call__(self, user_id: str, user_data: str, user_reviews: str, task_category: str, token_budget: int):
        """Obtain user summary from LLM"""

        prompt = f'''
        You are a real user on an online platform. Your user data and review history are as follows:
        {user_data}
        {user_reviews}
        Summarize your preferences and tastes using your information and past reviews into a summary that will help you rank items later.
        Your final output should be a paragraph with 300 tokens or less, and should not include any of your analysis process, only the summary of your preferences. 
        It should also be in the second person, with you instead of I. 
        Prioritize pertinent information focused on the task category of {task_category}, such as likes and dislikes, rather than trivial information.
        Your output should always start with the user_id and a colon.

        '''
        # print(prompt)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=token_budget
        )

        try:
            # print('Meta Output:',result)
            pattern = re.escape(user_id)
            match = re.search(pattern, reasoning_result)
            if match:
                self.memory(user_id, reasoning_result)
                # return reasoning_result
            else:
                raise Exception()
            # print('Processed Output:', eval(result))
            # time.sleep(4)
            return reasoning_result
        except:
            print("Bad LLM output")
            # return ['']
            # Retry with larger token budget on failure
            return self.__call__(user_id, user_id, user_data, user_reviews, task_category, token_budget * 2)
    
class ItemReasoning(ReasoningBase):
    """
    Module to reason and summarize about each item
    """

    def __init__(self, profile_type_prompt, memory, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)

    def __call__(self, item_id: str, item_info: str, item_reviews: str, task_category: str, token_budget: int):
        """Obtain item summary from LLM"""

        prompt = f'''
        You are a helpful assistant that summarizes details and sentiments given items and their reviews. Your item category is: {task_category}.
        Here is your item data: {item_info}.
        Your task is to take the data and the following reviews and output a summary of the item, including details on what type of item it is and general sentiment of it.
        Include details that will be useful for ranking items based on user preference.
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
            # print('Meta Output:',result)
            pattern = re.escape(item_id)
            match = re.search(pattern, reasoning_result)
            if match:
                self.memory(item_id, reasoning_result)
                # print(reasoning_result)
            else:
                raise Exception()
            # print('Processed Output:', eval(result))
            # time.sleep(4)
            return reasoning_result
        except:
            print("Bad LLM output")
            # return ['']
            # Retry with larger token budget on failure
            return self.__call__(item_id, item_info, item_reviews, task_category, token_budget * 2)

class TaskReasoning(ReasoningBase):
    """Module to reason about task"""

    def __init__(self, profile_type_prompt, memory, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)
        
    def __call__(self, task_description: str, token_budget: int):
        """Obtain ranking from LLM"""

        prompt = task_description
        reasoning_results = []

        # Get result from prompt
        messages = [{"role": "user", "content": prompt}]
        reasoning_results.append(
            self.llm(
                messages=messages,
                temperature=0.0,
                max_tokens=token_budget
            )
        )

        # Modify prompt with feedback from most similar completed tasks
        feedbacks = self.memory(task_description)
        for feedback in feedbacks:
            modified_prompt = prompt + feedback    
            messages = [{"role": "user", "content": modified_prompt}]
            reasoning_results.append(
                self.llm(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=token_budget
                )
            )
        print(reasoning_results)
        try:
            # print('Meta Output:',result)
            ranking_list = []
            confidence_scores = []
            # Try to convert all outputs to lists
            for reasoning_result in reasoning_results:
                try:
                    match = re.search(r"\[.*\]", reasoning_result, re.DOTALL)
                    if match:
                        result = match.group()
                        ranking = eval(result)
                        confidence_score = eval(reasoning_result)[1]
                        ranking_list.append(ranking)
                        confidence_scores.append(confidence_score)
                except:
                    continue
            # Only retry __call__ if all outputs are invalid
            if not ranking_list or not confidence_scores:
                raise Exception()
            # Do voting between all output lists
            final_ranking = weighted_borda_vote(ranking_list, confidence_scores)
            print('Processed Output:', final_ranking)
            # Average of all confidence scores
            confidence_score = sum(confidence_scores) / len(confidence_scores)
            # Recreate string output using voting results
            final_reasoning_result = str((final_ranking, confidence_score))
            print(final_reasoning_result)
            self.memory(task_description, task_description + final_reasoning_result)
            # time.sleep(4)
            return final_ranking
        except:
            print('format error')
            # return ['']
            # Retry with larger token budget on failure
            return self.__call__(task_description, token_budget * 2)
    
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

class TaskMemory(MemoryBase):
    def __init__(self, memory_type: str, llm) -> None:
        self.llm = llm
        self.embedding = self.llm.get_embedding_model()
        self.db_path = os.path.join('./db', memory_type)
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=self.db_path
        )

    def __call__(self, current_situation: str, trajectory: str = None):
        if not trajectory:
            return self.retriveMemory(current_situation)
        else:
            return self.addMemory(current_situation, trajectory)

    def retriveMemory(self, query_scenario: str):
        # Return empty string if memory is empty
        if self.scenario_memory._collection.count() == 0:
            return []
        
        # Similarity threshold
        threshold = 0.1

        # Find most similar memories
        task_name = query_scenario
        similarity_results = self.scenario_memory.similarity_search_with_score(task_name, k=1)

        # Extract task trajectories from results
        task_trajectories = []
        for doc, distance in similarity_results:
            # print(distance)
            if distance <= threshold:
                task_trajectories.append(doc.metadata['task_trajectory'])

        # Returns empty string if no tasks similar enough
        if not task_trajectories:
            return []
        
        # Use previous trajectories to refine prompt to current task
        responses = []
        for task_trajectory in task_trajectories:
            prompt = f'''
        You will be given a previous task that you completed and your output for that task.
        The output is in the format of a tuple, where the second value of the tuple is how confident you were in your output on a scale from 0-1.
        From both this task and your confidence score, analyze how you might reason about a similar task to this one.
        For example, if your confidence was low, you may want to try a different approach to solving the new task.
        If your confidence was high, you may want to use the same approach.
        Think of it as shaking up the approach when you aren't very sure of the answer.
        Here is the previous task and output:
        {task_trajectory}

        Keep your output to less than 300 tokens.
        Your output should be in the form of a list of reasoning steps that can help you in the future task like the below example.
        Here is a list of reasoning instructions that might help you:
        1.
        2.
        3.

        '''
            append_to_file(prompt, "output.txt")

            responses.append(
                self.llm(
                    messages=[{"role": "user", "content": prompt}], 
                    temperature=0.2, 
                    max_tokens=15000
                )
            )
        # response = None
        return responses

    def addMemory(self, current_situation: str, trajectory: str):
        task_name = current_situation
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": trajectory
            }
        )
        self.scenario_memory.add_documents([memory_doc])

    def deleteData(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
    
class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of RecommendationAgent
    """

    llm = GeminiLLM(api_key="")
    user_memory = ModuleMemory(memory_type="user", llm=llm)
    item_memory = ModuleMemory(memory_type="item", llm=llm)
    task_memory = TaskMemory(memory_type="task", llm=llm)

    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.user_reasoning = UserReasoning(profile_type_prompt='', memory=MyRecommendationAgent.user_memory, llm=self.llm)
        self.item_reasoning = ItemReasoning(profile_type_prompt='', memory=MyRecommendationAgent.item_memory, llm=self.llm)
        self.task_reasoning = TaskReasoning(profile_type_prompt='', memory=MyRecommendationAgent.task_memory, llm=self.llm)

    def workflow(self, token_budget: int = 15000):
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

        user_id = ''
        user = ''
        item_list = []
        history_review = ''
        for sub_task in plan:
            if 'user' in sub_task['description']:
                user_id = self.task['user_id']
                user = str(self.interaction_tool.get_user(user_id=user_id))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])
            elif 'item' in sub_task['description']:
                for n_bus in range(len(self.task['candidate_list'])):
                    item = self.interaction_tool.get_item(item_id=self.task['candidate_list'][n_bus])
                    keys_to_extract = ['item_id', 'name','stars','review_count','attributes', 'categories', 'title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                    item_list.append(filtered_item)
            elif 'review' in sub_task['description']:
                history_review = str(self.interaction_tool.get_reviews(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass
        
        task_category = self.task['candidate_category']

        # Check if user summary already in memory
        result = self.user_memory.retriveMemory(user_id)
        if result:
            user_summary = result
            # print(user_summary)
            # print(result)
            # print(f"Tokens: {num_tokens_from_string(user_summary)}")
            # print(f"Tokens: {num_tokens_from_string(result)}")
        else:
            user_summary = self.user_reasoning(user_id, user, history_review, task_category, token_budget)

        user_description = f'''
        You are a real user on an online platform. Your user id and a summary of your user profile and review history are as follows:
        {user_summary}
        '''

        # Check if item summaries already in memory
        item_summaries = []
        for i in range(len(item_list)):
            item_id = self.task['candidate_list'][i]
            result = self.item_memory.retriveMemory(item_id)
            if result:
                item_summaries.append(result)       
                # print(item_list[i])  
                # print(result)
                # print(f'Item {i + 1} in memory')
                # print(result)
            else:                
                reviews = str(self.interaction_tool.get_reviews(item_id=item_id)[:25])
                item_summaries.append(self.item_reasoning(item_id, item_list[i], reviews, task_category, token_budget))

        task_description = f'''
        Now you will be given the following 20 items: {self.task['candidate_list']}.
        Your goal is to rank the items by how well they match to the user profile.
        Of these items, there is 1 ground truth item that is the true number 1 recommendation.
        The most important thing, and how you will be evaluated, is how highly you rank this ground-truth.
        In the ranking, make sure that the better matches are at the front of the list.
        In your process, prioritize the rating and sentiment of the candidates higher than the user's preferences.
        The raw data of the above 20 candidate items is as follows: {item_list}.
        Summaries of the candidates and sentiment surrounding them are as follows: {item_summaries}.
        Your final output should be ONLY a tuple with (ranked item list of candidates, confidence score of your decision from 0-1).
        Be strict with your confidence score, only give high scores for when you are very confident you have found the correct ground-truth.
        DO NOT introduce any other item ids! DO NOT output your analysis process!
        The correct output format:
        (['item id1', 'item id2', 'item id3', ...], confidence score)
        '''
        
        full_task = user_description + task_description
        # print(full_task)
        # full_task = user_description + task_description
        # print(full_task)
        # with open("output.txt", "a") as f:
        #     print(full_task, file=f)
        result = self.task_reasoning(full_task, token_budget)
        # print(result)
        # print(result)
        return result

if __name__ == "__main__":
    task_set = "yelp" # "goodreads" or "yelp"
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
    agent_outputs = simulator.run_simulation(number_of_tasks=100, enable_threading=True, max_workers=10)

    # Remove task memory
    MyRecommendationAgent.task_memory.deleteData()

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
