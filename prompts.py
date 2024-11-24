from langchain.prompts import PromptTemplate

assistant_template = """
You are an comic assistant chatbot named "xkcd-bot". Your expertise is exclusively in providing information and
advice about anything related to xkcd comics. If a question is not about comics,
respond with, "I specialize only in comic related queries."
Chat History: {chat_history}
Question: {question}
Answer:"""

assistant_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=assistant_template
)

api_url_template = """
Given the following API Documentation for xkcd's official API: {api_docs}
Your task is to construct the most efficient API URL to answer the user's question, ensuring the 
call is optimized to include only necessary information.
Question: {question}
API URL:
"""
api_url_prompt = PromptTemplate(input_variables=['api_docs', 'question'],
                                template=api_url_template)

api_response_template = """"
With the API Documentation for xkcd's official API: {api_docs} and the specific user question: {question} in mind,
and given this API URL: {api_url} for querying, here is the response from xkcd's API: {api_response}. 
Please provide a summary that directly addresses the user's question, 
omitting technical details like response format, and focusing on delivering the answer with clarity and conciseness, 
as if xkcd itself is providing this information.
Summary:
"""
api_response_prompt = PromptTemplate(input_variables=['api_docs', 'question', 'api_url',
                                                      'api_response'],
                                     template=api_response_template)
