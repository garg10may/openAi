from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from utility import load_key


def csv_agent():
  agent = create_csv_agent(OpenAI(temperature=0),
                              'employee_data.csv',
                              verbose=True)
  return agent

prompt = '''
You are working with a pandas dataframe in Python. The name of the
dataframe is `df`. You should use tolls below to answer the question posed to you:

python_repl_ast: A Python shell. Use this to execute python commands. Input should be 
a valid python command. When using this tool, sometimes output is abbreviated = make sure it does 
not look abbreviated before using it in your answer. 

Use the following format:

Question: The input question you must answer
Thought: you should always think about what to do 
Action: The action to take, should be one of 
[python_repl_ast]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question 
'''
if __name__ == '__main__':
  agent = csv_agent()
  print(agent)