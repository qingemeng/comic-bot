from langchain_openai import OpenAI
from langchain.chains import LLMChain, APIChain
from prompts import assistant_prompt, api_response_prompt, api_url_prompt
from langchain.memory.buffer import ConversationBufferMemory
from dotenv import load_dotenv

api_docs = """
openapi: 3.0.0

info:
  version: 1.0.0
  title: xkcd
  description: 'A webcomic of romance, sarcasm, math, and language.'

servers:
  - url: https://xkcd.com/
    description: Official xkcd JSON interface

paths:
  # Retrieve the current comic
  /info.0.json:
    get:
      # A list of tags to logical group operations by resources and any other
      # qualifier. 
      tags:
        - comic
      description: Returns comic based on ID
      summary: Find latest comic
      # Unique identifier for the operation, tools and libraries may use the
      # operationId to uniquely identify an operation.
      operationId: getComic
      responses:
        '200':
          description: Successfully returned a comic
          content:
            application/json:
              schema:
                # Relative reference to prevent duplicate schema definition.
                $ref: '#/components/schemas/Comic'
  # Retrieve a comic by ID
  /{id}/info.0.json:
    get:
      tags:
        - comic
      description: Returns comic based on ID
      summary: Find comic by ID
      operationId: getComicById
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Successfully returned a commmic
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Comic'

components:
  schemas:
    Comic:
      type: object
      properties:
        month:
          type: string
        num:
          type: integer
        link:
          type: string
        year:
          type: string
        news:
          type: string
        safe_title:
          type: string
        transcript:
          type: string
        alt:
          type: string
        img:
          type: string
        title:
          type: string
        day:
          type: string
          """

import chainlit as cl

load_dotenv()


@cl.on_chat_start
def setup_multiple_chains():
  llm = OpenAI(model='gpt-3.5-turbo-instruct',
               temperature=0)
  conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                 max_len=200,
                                                 return_messages=True,
                                                 )
  llm_chain = LLMChain(llm=llm, prompt=assistant_prompt, memory=conversation_memory)
  cl.user_session.set("llm_chain", llm_chain)

  api_chain = APIChain.from_llm_and_api_docs(
      llm=llm,
      api_docs=api_docs,
      api_url_prompt=api_url_prompt,
      api_response_prompt=api_response_prompt,
      verbose=True,
      limit_to_domains=["https://xkcd.com"]
  )
  cl.user_session.set("api_chain", api_chain)


@cl.on_message
async def handle_message(message: cl.Message):
  user_message = message.content.lower()
  llm_chain = cl.user_session.get("llm_chain")
  api_chain = cl.user_session.get("api_chain")

  if any(keyword in user_message for keyword in ["comic"]):
    # If any of the keywords are in the user_message, use api_chain
    response = await api_chain.acall(user_message,
                                     callbacks=[cl.AsyncLangchainCallbackHandler()])
  else:
    # Default to llm_chain for handling general queries
    response = await llm_chain.acall(user_message,
                                     callbacks=[cl.AsyncLangchainCallbackHandler()])

  response_key = "output" if "output" in response else "text"
  await cl.Message(response.get(response_key, "")).send()
