from typing import Dict, TypedDict, List
from langgraph.graph import END, StateGraph
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
import brickschema
import rdflib
import os
from utils import *
import json
import re
from sentence_transformers import SentenceTransformer, util
import openai
# Custom exception for timeout
import multiprocessing
import time
# Disable the huggingface tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def run_query(query, g):
    """Runs the SPARQL query."""
    res = g.query(query)
    result_df = extract_query_result_as_dataframe(res)
    return result_df


def extract_query_result_as_dataframe(sparql_result):
    """
    Extracts SPARQL query results into a DataFrame.

    Args:
    - sparql_result (SPARQLResult): Result of the SPARQL query.

    Returns:
    - DataFrame: Each row of the DataFrame represents a row from the query result.
    """
    # Extract variable names from SPARQLResult
    var_names = [str(var) for var in sparql_result.vars]

    # Initialize list to store results
    result_list = []

    # Iterate through SPARQLResult and extract values
    for row in sparql_result:
        result_list.append([str(cell.split("#")[-1]) if isinstance(cell, rdflib.term.URIRef) else str(cell) for cell in row])

    # Convert result list into a DataFrame
    df = pd.DataFrame(result_list, columns=var_names)

    return df
    
def BuildingQA(question, hvac_queries, expt_llm, g, max_iterations, data_dir):

    def add_tokens(prompt_tokens, completion_tokens, total_tokens, cost, cb):
        prompt_tokens += cb.prompt_tokens
        completion_tokens += cb.completion_tokens
        total_tokens += cb.total_tokens
        cost += cb.total_cost
        return prompt_tokens, completion_tokens, total_tokens, cost
    mpnet_v2 = SentenceTransformer('all-mpnet-base-v2')

    def find_similar_labels(question, descriptions, hvac_queries, threshold=0.1):
        # Encode the question into an embedding
        ques_embedding = mpnet_v2.encode(question, convert_to_tensor=True)

        similarity_scores = []
        
        for desc in descriptions:
            # Encode the description into an embedding
            desc_embedding = mpnet_v2.encode(desc, convert_to_tensor=True)
            
            # Compute similarity score
            similarity_score = util.pytorch_cos_sim(ques_embedding, desc_embedding).item()
            similarity_scores.append((desc, similarity_score))

        # Sort descriptions by similarity score in descending order
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top 3 descriptions if they exceed the threshold
        top_3_desc = [desc for desc, score in similarity_scores[:3] if score >= threshold]

        # Generate formatted output
        result_string = ""
        for desc in top_3_desc:
            query = hvac_queries.get(desc, "No SPARQL query found.")
            result_string += f'Description: "{desc}"\nSPARQL: "{query}"\n{"-"*30}\n'
        
        print(result_string)  # Print the result
        return result_string  # Return the formatted string if needed

    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            query_result : explanation of the query result
            messages : With user question, error messages, reasoning
            classes : list of classes that should be queried
            iterations : Number of tries
            human_feedback: Human feedback
        """

        messages: List
        question: str
        final_query: str
        query_result: str
        query_success: bool
        iterations: int
        ts_df: pd.DataFrame
        ts_ref: pd.DataFrame
        data_success: bool
        ts_string: str
        func_called: str
        prompt_tokens: int
        total_tokens: int
        completion_tokens: int
        cost: float

    def text2sparql(state: GraphState):

        print("------TEXT2SPARQL-----")
        question=state["question"]
        prompt_tokens=0
        total_tokens=0
        completion_tokens=0
        cost=0
        descriptions = list(hvac_queries.keys())
        few_shot_string=find_similar_labels(question, descriptions, hvac_queries, threshold=0.1)
        # Grader prompt
        query_generate_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", """You are a great Brick Schema Expert, specialized in writing SPARQL queries to extract data needed for a question. 
                    Remember that your SPARQL query are not expected to answer the question directly, it will only extract the data needed to answer the question.
                    Thus, aim at only writing the SPARQL query, and not the final answer to the question.
                    
                Here are some example queries relevant to the question at hand: "{few_shot_string}"
                ### Constraints & Instructions:
                - use the prefix PREFIX brick: <https://brickschema.org/schema/Brick#>
                - make sure your answer is a directly executable SPARQL query"""
                ),
                ("placeholder", "{messages}"),
            ]
        )


        # Data model
        class QueryCode(BaseModel):
            """Code output"""
            query: str = Field(description="the SPARQL query to extract the time series data")


        llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('CBORG_API_KEY'), openai_api_base="https://api.cborg.lbl.gov" ,model=expt_llm)
        query_generate_chain = query_generate_prompt | llm.with_structured_output(QueryCode)
        with get_openai_callback() as cb:
            solution = query_generate_chain.invoke({"messages":[("user",question)],"few_shot_string":few_shot_string})
            prompt_tokens, completion_tokens, total_tokens, cost = add_tokens(prompt_tokens, completion_tokens, total_tokens, cost, cb)
        query=solution.query

        recall_llm=False
        i=0
        while i < max_iterations:

            if recall_llm:
                llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('CBORG_API_KEY'), openai_api_base="https://api.cborg.lbl.gov" ,model=expt_llm)
                query_generate_chain = query_generate_prompt | llm.with_structured_output(QueryCode)
                with get_openai_callback() as cb:
                    solution = query_generate_chain.invoke({"messages":[("user",question), ("user",f"Previous query you have written: {query}, failed with the error: {error}")],"few_shot_string":few_shot_string})
                    prompt_tokens, completion_tokens, total_tokens, cost = add_tokens(prompt_tokens, completion_tokens, total_tokens, cost, cb)
                query=solution.query


            try :
                result_df = run_query(query, g)
                print(result_df.head())
                #convert the result_df into a string format. 
                result_string=result_df.head().to_string(index=False)
                return  {"final_query": query, "ts_ref":result_df, "query_result": result_string, "query_success": True, "prompt_tokens":prompt_tokens, "total_tokens":total_tokens, "completion_tokens":completion_tokens, "cost":cost} 
                
            except Exception as e:
                error=e
                print(e)
                i += 1
                print(f"Query failed. Retrying. Iteration {i}")
                recall_llm=True
                continue

            
        return  {"final_query": None, "ts_ref":None,"query_result": None, "query_success": False, "prompt_tokens":prompt_tokens, "total_tokens":total_tokens, "completion_tokens":completion_tokens, "cost":cost} 
    

    def extract_data(state: GraphState):
        print("------EXTRACT DATA-----")

        question=state["question"]
        concatenated_content = state["query_result"]
        result_df = state["ts_ref"]

        prompt_tokens=state["prompt_tokens"]
        total_tokens=state["total_tokens"]
        completion_tokens=state["completion_tokens"]
        cost=state["cost"]
        # Grader prompt
        data_get_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", """You are a coding assistant with expertise in extracting timeseries data. 
                    You have access to a database containing time series data from a buildings HVAC system, collected from various sensors. 
                    Your task is to write a function that will extract this data. 
                    You are not expected to answer the question. Only extract the timeseries data.

                ### Variables:
                1. **`result_df`**: A Pandas DataFrame with time series references that looks like this: {context}
                2. **`data_dir`**: A string representing the path to the directory containing time series data files.

                ### Constraints & Instructions:
                - Do not assign values to `result_df` or `data_dir', use them as provided.
                - Your code will first extract the timeseries references from `result_df`, using the column shown in the example.
                - Then, you will construct file paths as: `data_dir/<file_name>.csv`.
                - Name your function extract_timeseries_data, it will only take result_df and data_dir as input, and it will return the dataframe read from the file.

                - Structure your output as a JSON object with two keys:
                - `'imports'`: Required Python imports.
                - `'code'`: The complete executable script.""",
                ),
                ("placeholder", "{messages}"),
            ]
        )


        # Data model
        class codeOutput(BaseModel):
            """Code output"""
            imports: str = Field(description="Code block import statements")
            code: str = Field(description="Function called extract_timeseries_data with two inputs: result_df and data_dir")
            description = "Schema for code solutions to questions regarding data collected from a building."


        expt_llm = "openai/gpt-4o"
        llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('CBORG_API_KEY'), openai_api_base="https://api.cborg.lbl.gov" ,model=expt_llm)
        data_get_chain = data_get_prompt | llm.with_structured_output(codeOutput)

        with get_openai_callback() as cb:
            solution = data_get_chain.invoke({"context":concatenated_content, "messages":[("user",question)]})
            prompt_tokens, completion_tokens, total_tokens, cost = add_tokens(prompt_tokens, completion_tokens, total_tokens, cost, cb)

        
        
        imports=solution.imports
        code=solution.code
        recall_llm=False
        i=0
        while i < max_iterations:
            if recall_llm:
                llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('CBORG_API_KEY'), openai_api_base="https://api.cborg.lbl.gov" ,model=expt_llm)
                data_get_chain = data_get_prompt | llm.with_structured_output(codeOutput)
                with get_openai_callback() as cb:
                    solution = data_get_chain.invoke({"context":concatenated_content, "messages":[("user",question), ("user",f"Previous code you have written: {code}, failed with the error: {error}")]})
         
                    prompt_tokens, completion_tokens, total_tokens, cost = add_tokens(prompt_tokens, completion_tokens, total_tokens, cost, cb)

                imports=solution.imports
                code=solution.code    
 

            try:
                print(imports + "\n" + code)
                exec(imports + "\n" + code, globals())
                sup_df=extract_timeseries_data(result_df, data_dir)
                #convert the result_df into a string format. 
                sup_content=sup_df.head().to_string(index=False)
                print(sup_content)
                return {"ts_df":sup_df, "ts_string":sup_content,  "data_success":True, "prompt_tokens":prompt_tokens, "total_tokens":total_tokens, "completion_tokens":completion_tokens, "cost":cost}
            except Exception as e:
                error=e
                print(e)
                i += 1
                print(f"Code failed. Retrying. Iteration {i}")
                recall_llm=True
                continue

        return {"ts_df":sup_df, "ts_string":sup_content, "data_success":False, "prompt_tokens":prompt_tokens, "total_tokens":total_tokens, "completion_tokens":completion_tokens, "cost":cost}

    def router(state: GraphState):

        print("------ROUTER-----")
        prompt_tokens=state["prompt_tokens"]
        total_tokens=state["total_tokens"]
        completion_tokens=state["completion_tokens"]
        cost=state["cost"]
        router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", """You are an agent that decides on which function to call based on the requirements of a given question regarding HVAC systems.

                
                If this question indicates the need for a computation, you will return a string named "compute_timeseries_analysis"
                if this question indicates that the user wants a visualization such as a plot, you will return a string named "plot_data". 
                Do not call any other function rather than these two.  """,
                ),
                ("placeholder", "{messages}"),
            ]
        )

        # Data model
        class RouterOutput(BaseModel):
            """Code output"""
            func_called: str = Field(description="name of the function to be called")

        print("calling llm at router")

        llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('CBORG_API_KEY'), openai_api_base="https://api.cborg.lbl.gov" ,model=expt_llm)
        router_chain = router_prompt | llm.with_structured_output(RouterOutput)
        
        with get_openai_callback() as cb:
            solution = router_chain.invoke({"messages":[("user",question)]})
            prompt_tokens, completion_tokens, total_tokens, cost = add_tokens(prompt_tokens, completion_tokens, total_tokens, cost, cb)


        print("solution:",solution)
        print("Router:",solution.func_called)
        return {"func_called":solution.func_called, "prompt_tokens":prompt_tokens, "total_tokens":total_tokens, "completion_tokens":completion_tokens, "cost":cost}

    def func_router(state: GraphState):
        """
        Checking if all queries returned no results.
        Then, the query writer is called again.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        print("---FUNC_ROUTER---")
        func_called=state["func_called"]
        query_success=state["query_success"]
        if func_called=="compute_timeseries_analysis":
            return "computation_code_gen"
        elif func_called=="plot_data":
            return "plot_code_gen"
        
        if query_success==False:
            return "end"
        else:
            return "extract_data"




    def computation_code_gen(state: GraphState):


        print("------COMPUTATION CODE GENERATION-----")

        ts_df=state["ts_df"]
        ts_string=state["ts_string"]
        prompt_tokens=state["prompt_tokens"]
        total_tokens=state["total_tokens"]
        completion_tokens=state["completion_tokens"]
        cost=state["cost"]

        computation_code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", """You are a coding assistant with expertise in time series data analysis. 
                    You have access to a dataframe with time series values from a buildings HVAC system.
                    Your task is to write a function that will do some computation on the data, and print the results, based on the question.

                ### Inputs:
                1. **`ts_df`**: A Pandas DataFrame with time series values that looks like this: {context}

                ### Constraints & Instructions:
                - Do not assign values to `ts_df`
                - Your code will first extract the timeseries values from `ts_df`, based on columns shown in the example.
                - if the time column is not shown/results in an error, assume the index is the time column.
                - Then, it will filter if there is a notion of time in the question (i.e., last month). Make sure to convert the time series values to datetime objects.
                - When filtering, assume the latest timestamp in the dataset represents the **current time**, and go backwards from there.
                - Name your function computation_data, it will only take ts_df as input,
                  and it will print the result of the computation needed to answer the question. 
                  Print statement will be understandable to a human with the necessary details.

                - Structure your output as a JSON object with two keys:
                - `'imports'`: Required Python imports.
                - `'code'`: The complete executable script.""",
                ),
                ("placeholder", "{messages}"),
            ]
        )


        # Data model
        class codeOutput(BaseModel):
            """Code output"""
            imports: str = Field(description="Code block import statements")
            code: str = Field(description="Function called computation_data that conducts the necessary computation on the time series data called ts_df")
            description = "Schema for code solutions to questions regarding data collected from a building."


        expt_llm = "openai/gpt-4o"
        llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('CBORG_API_KEY'), openai_api_base="https://api.cborg.lbl.gov" ,model=expt_llm)
        computation_code_gen_chain = computation_code_gen_prompt | llm.with_structured_output(codeOutput)
        with get_openai_callback() as cb:
            solution = computation_code_gen_chain.invoke({"context":ts_string, "messages":[("user",question)]})
            prompt_tokens, completion_tokens, total_tokens, cost = add_tokens(prompt_tokens, completion_tokens, total_tokens, cost, cb)



       
        
        print(solution)
                
        imports=solution.imports
        code=solution.code
        recall_llm=False
        i=0
        while i < max_iterations:
            if recall_llm:
                llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('CBORG_API_KEY'), openai_api_base="https://api.cborg.lbl.gov" ,model=expt_llm)
                computation_code_gen_chain = computation_code_gen_prompt | llm.with_structured_output(codeOutput)
                with get_openai_callback() as cb:
                        solution = computation_code_gen_chain.invoke({"context":ts_string, "messages":[("user",question), ("user",f"Previous code you have written: {code}, failed with the error: {error}, Rewrite it!")]})
                        prompt_tokens, completion_tokens, total_tokens, cost = add_tokens(prompt_tokens, completion_tokens, total_tokens, cost, cb)

                imports=solution.imports
                code=solution.code    


            try:
                print(imports + "\n" + code)
                exec(imports + "\n" + code, globals())
                computation_data(ts_df)
                #convert the result_df into a string format. 
                return {"prompt_tokens":prompt_tokens, "total_tokens":total_tokens, "completion_tokens":completion_tokens, "cost":cost}
            except Exception as e:
                error=e
                print(e)
                i += 1
                print(f"Code failed. Retrying. Iteration {i}")
                recall_llm=True
                continue

        return {"prompt_tokens":prompt_tokens, "total_tokens":total_tokens, "completion_tokens":completion_tokens, "cost":cost}

    def plot_code_gen(state: GraphState):
        print("------PLOT CODE GENERATION-----")

        ts_df=state["ts_df"]
        ts_string=state["ts_string"]
        prompt_tokens=state["prompt_tokens"]
        total_tokens=state["total_tokens"]
        completion_tokens=state["completion_tokens"]
        cost=state["cost"]
        # Grader prompt
        plot_code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", """You are a coding assistant with expertise in time series data analysis. 
                    You have access to a dataframe with time series values from a buildings HVAC system.
                    Your task is to write a function that will make a plot of the data, based on the question.

                ### Inputs:
                1. **`ts_df`**: A Pandas DataFrame with time series values that looks like this: {context}

                ### Constraints & Instructions:
                - Do not assign values to `ts_df`
                - Your code will first extract the timeseries values from `ts_df`, based on columns shown in the example.
                - Then, it will filter if there is a notion of time in the question (i.e., last month). Make sure to convert the time series values to datetime objects.
                - When filtering, assume the latest timestamp in the dataset represents the **current time**, and go backwards from there.
                - Name your function plot_data, it will only take ts_df as input, and it will return the plot. 

                - Structure your output as a JSON object with two keys:
                - `'imports'`: Required Python imports.
                - `'code'`: The complete executable script.""",
                ),
                ("placeholder", "{messages}"),
            ]
        )


        # Data model
        class codeOutput(BaseModel):
            """Code output"""
            imports: str = Field(description="Code block import statements")
            code: str = Field(description="Function called plot_data that conducts the necessary computation on the time series data called ts_df")
            description = "Schema for code solutions to questions regarding data collected from a building."


        expt_llm = "openai/gpt-4o"

        llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('CBORG_API_KEY'), openai_api_base="https://api.cborg.lbl.gov" ,model=expt_llm)
        plot_code_gen_chain = plot_code_gen_prompt | llm.with_structured_output(codeOutput)
        
        with get_openai_callback() as cb:
            solution = plot_code_gen_chain.invoke({"context":ts_string, "messages":[("user",question)]})
            prompt_tokens, completion_tokens, total_tokens, cost = add_tokens(prompt_tokens, completion_tokens, total_tokens, cost, cb)


        print(solution)
                
        imports=solution.imports
        code=solution.code
        recall_llm=False
        i=0
        while i < max_iterations:
            if recall_llm:
                llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('CBORG_API_KEY'), openai_api_base="https://api.cborg.lbl.gov" ,model=expt_llm)
                computation_code_gen_chain = plot_code_gen_prompt | llm.with_structured_output(codeOutput)


                with get_openai_callback() as cb:
                    solution = computation_code_gen_chain.invoke({"context":ts_string, "messages":[("user",question), ("user",f"Previous code you have written: {code}, failed with the error: {error}, Rewrite it!")]})

                    prompt_tokens, completion_tokens, total_tokens, cost = add_tokens(prompt_tokens, completion_tokens, total_tokens, cost, cb)


                imports=solution.imports
                code=solution.code    


            try:
                print(imports + "\n" + code)
                exec(imports + "\n" + code, globals())
                plot_data(ts_df)

                #ts_df might have been modified.
                ts_string=ts_df.head().to_string(index=False)
                #convert the result_df into a string format. 
                return {"ts_string":ts_string, "ts_df":ts_df, "prompt_tokens":prompt_tokens, "total_tokens":total_tokens, "completion_tokens":completion_tokens, "cost":cost}
            except Exception as e:
                error=e
                print(e)
                i += 1
                print(f"Code failed. Retrying. Iteration {i}")
                recall_llm=True
                continue

        return {"ts_string":ts_string, "ts_df":ts_df, "prompt_tokens":prompt_tokens, "total_tokens":total_tokens, "completion_tokens":completion_tokens, "cost":cost}





    workflow = StateGraph(GraphState)

    """
    workflow.add_node("text2sparql", text2sparql)
    workflow.add_node("extract_data", extract_data)
    workflow.add_edge("text2sparql","extract_data")
    workflow.add_node("router", router)
    workflow.add_edge("extract_data","router")
    workflow.add_node("plot_code_gen", plot_code_gen)
    workflow.add_node("computation_code_gen", computation_code_gen)

    workflow.add_edge("plot_code_gen", END)
    workflow.add_edge("computation_code_gen", END)
 

    workflow.add_conditional_edges(
        "router",
        func_router,
        {
            "computation_code_gen": "computation_code_gen",
            "plot_code_gen": "plot_code_gen",
            "extract_data": "extract_data",
            "end": END,
        },
    )
    """
    workflow.add_node("text2sparql", text2sparql)
    workflow.add_node("extract_data", extract_data)
    workflow.add_edge("text2sparql","extract_data")
    workflow.add_edge("extract_data","plot_code_gen")
    workflow.add_node("plot_code_gen", plot_code_gen)
    workflow.add_node("computation_code_gen", computation_code_gen)
    workflow.add_edge("plot_code_gen","computation_code_gen")
    workflow.add_edge("computation_code_gen", END)
    # Build graph
    workflow.set_entry_point("text2sparql")

    app = workflow.compile()
    solution = app.invoke({"question":question,"iterations": 0, "promt_tokens":0, "total_tokens":0, "completion_tokens":0, "cost":0})  

    return solution
