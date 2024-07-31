import os
import time
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_search import YoutubeSearch
from fpdf import FPDF
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import initialize_agent, Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_react_agent, tool
from flask import Flask, request, jsonify
import chromadb
import sqlite3
import re
import textwrap
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain.chains.llm import LLMChain
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import word_tokenize
import pytube
from moviepy.editor import *

# Download necessary resources
nltk.download('punkt')



# Initialize environment variables
from dotenv import load_dotenv
import traceback
import logging

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_TOKEN')
YT_API_KEY = os.getenv('YT_API_KEY')

LANGCHAIN_TRACING_V2='true'
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT="default"

# Download and initialize all required models
model = SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
summarization_model_name = "suriya7/bart-finetuned-text-summarization"
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)


# Function to load the vector database
def load_vectordb():
    """
    Load the vector database from Chroma.

    Returns:
        langchain_chroma (Chroma): The Chroma vector database.
    """
    persistent_client = chromadb.PersistentClient("./chromadb")

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="knowledge_base",
        embedding_function=model,
    )

    return langchain_chroma

vector_db = load_vectordb()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_execute(func, *args, **kwargs):
    """
    Execute a function safely, catching any exceptions and logging errors.

    Args:
        func (callable): The function to execute.
        *args: Variable length argument list for the function.
        **kwargs: Arbitrary keyword arguments for the function.

    Returns:
        The result of the function execution, or an error message if an exception occurs.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        return f"An error occurred: {str(e)}"


# Initialize LLM
llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo-16k")


def count_tokens(text):
    """
    Count the number of tokens in a given text using NLTK's word tokenizer.

    Args:
        text (str): The input text.

    Returns:
        int: The number of tokens in the text.
    """
    tokens = word_tokenize(text)
    return len(tokens)

def text_summarize(text):
    """
    Summarize the input text using a MapReduce approach.

    Args:
        text (str): The input text to summarize.

    Returns:
        str: The summary of the input text.
    """
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)

    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

    # Map step
    map_template = """The following is a document:
    {docs}
    Based on this document, please identify the main themes and key points.
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce step
    reduce_template = """The following is a set of summaries:
    {docs}
    Take these and distill them into a final, consolidated summary of the main themes and key points.
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combine
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="docs"
    )

    # Create the MapReduceDocumentsChain
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=combine_documents_chain,
        document_variable_name="docs"
    )

    return map_reduce_chain.run(docs)


# Function to add documents to the database
def add_documents_to_db(pdf_file):
    """
    Add documents extracted from a PDF file to the vector database.

    Args:
        pdf_file (str): The path to the PDF file to process.
    """
    try:
        texts = extract_text_from_pdf(pdf_file)
        cleaned_text = clean_text(texts)
        documents = get_text_chunks(cleaned_text)
        
        if documents:
            h_size = 10000
            total_documents = len(documents)
            processed_documents = 0

            while processed_documents < total_documents:
                remaining_documents = total_documents - processed_documents
                current_h_size = min(h_size, remaining_documents)

                h_documents = documents[processed_documents:processed_documents + current_h_size]
                vector_db.add_documents(h_documents)

                processed_documents += current_h_size

                print(f"Processed {processed_documents} out of {total_documents} documents.")

            print("All documents added to the collection.")
        else:
            logger.warning(f"No documents found in {pdf_file}.")
    except Exception as e:
        logger.error(f"Error adding documents to database from {pdf_file}: {str(e)}")
        raise  # Re-raise the exception for visibility


def generate_valid_filename(query):
    """
    Generate a valid filename by replacing invalid characters with underscores.

    Args:
        query (str): The input string to generate the filename from.

    Returns:
        str: The generated valid filename.
    """
    valid_chars = '-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    filename = ''.join(c if c in valid_chars else '_' for c in query)
    return filename

#################################################
##              NEW FUNCTIONS                  ##
#################################################
import whisper
import time
from pytubefix import YouTube
from pytubefix.cli import on_progress


def download_video_mp3(url):
    yt = YouTube(url, on_progress_callback = on_progress) 
    ys = yt.streams.get_audio_only()
    file = ys.download(mp3=True)

    return file

def audio_to_text(filename):
    
    model = whisper.load_model("tiny")
    result = model.transcribe(filename)

    transcription = result["text"]

    return transcription


#################################################
# Function to search and transcribe YouTube videos
def search_and_transcribe_videos(query, max_results=20, min_valid_videos=4):
    """
    Search for YouTube videos and transcribe them.

    Args:
        query (str): The search query for YouTube videos.
        max_results (int): The maximum number of results to fetch. Default is 20.
        min_valid_videos (int): The minimum number of valid videos to transcribe. Default is 4.

    Returns:
        str: The path to the transcript file.
    """
    valid_urls = []
    current_max_results = max_results
    transcription = ''
    while len(valid_urls) < min_valid_videos and current_max_results <= 20:
        results = YoutubeSearch(query, max_results=current_max_results).to_dict()
        filtered_results = [video for video in results if video.get('liveBroadcastContent') != 'live']
        for video in filtered_results:
            video_id = video['id']
            video_link = f"https://www.youtube.com/watch?v={video_id}"
            try:
                transcription = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
                transcript_text = " ".join([line['text'] for line in transcription])
                valid_urls.append((transcript_text))

            except:
              continue

            if len(valid_urls) >= min_valid_videos:
                 break

    current_max_results += max_results

    transcript_file = generate_valid_filename(query) + '.txt'
    with open(transcript_file, 'a', encoding='utf-8') as f:
      for text in valid_urls[:min_valid_videos]:
        f.write(f"Text:{text}\n\n")
    
    return transcript_file

# Function to create a PDF from a transcript
def create_pdf(input_file):
    """
    Create a PDF file from a transcript file.

    Args:
        input_file (str): The path to the transcript file.

    Returns:
        str: The path to the created PDF file.
    """
    pdf = FPDF()
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))
    filename = input_file.split('.txt')[0]
    output_filename = f"{filename}.pdf"
    pdf.output(output_filename)
    return output_filename

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): The path to the PDF file.
    
    Returns:
        str: The extracted text.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to clean extracted text
def clean_text(text):
    """
    Clean and preprocess the extracted text.
    
    Args:
        text (str): The extracted text.
    
    Returns:
        str: The cleaned text.
    """

    text = text.replace('\xa0', ' ')
    text = re.sub(r'[^\x00-\x7F]+!?', ' ', text)
    return text

# Function to split text into chunks
def get_text_chunks(text):
    """
    Split the cleaned text into manageable chunks for further processing.
    
    Args:
        text (str): The cleaned text.
        chunk_size (int): The size of each text chunk.
    
    Returns:
        list of Document: List of Document objects containing text chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]



# Function to process YouTube videos
def load_video(url):
    """
    Retrieve the transcript of a YouTube video, save it to a text file, 
    convert the text file to a PDF, and return the PDF filename.
    
    Args:
        url (str): The URL of the YouTube video.
    
    Returns:
        str: The filename of the generated PDF.
    """
    video_id = url.split('v=')[-1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = ' '.join([t['text'] for t in transcript])
    filename = f"{video_id}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    pdf_filename = create_pdf(filename)
    return pdf_filename

#Initialize the collection 
def initialize_collection():
    """
    Initialize the knowledge base by searching and transcribing YouTube videos 
    for a predefined set of queries, converting them to PDF, and adding them 
    to the vector database.
    
    Returns:
        bool: True if the initialization is successful.
    """
    # Update queries if you want the assistant to have a different knowledge base and uncomment initialize_collection() after this function
    
    queries = [
        "Transfer Learning in Machine Learning",
        "Object Detection and Recognition in Computer Vision",
        "Sentiment Analysis in Natural Language Processing",
        "Generative Adversarial Networks (GANs) in Deep Learning",
        "Automatic Speech Recognition (ASR) Systems",
        "Reinforcement Learning Applications",
        "Image Segmentation Techniques in Computer Vision",
        "Text Summarization Methods in NLP",
        "Convolutional Neural Networks (CNNs) for Image Classification",
        "Speech Synthesis and Text-to-Speech (TTS) Systems",
        "Anomaly Detection in Machine Learning",
        "Facial Recognition Technology and Ethics",
        "Machine Translation and Language Models",
        "Recurrent Neural Networks (RNNs) for Sequence Data",
        "Speaker Diarization and Identification in Speech Processing",
        "Applications of Natural Language Understanding (NLU)",
        "Deep Reinforcement Learning for Game AI",
        "Semantic Segmentation in Computer Vision",
        "Dialogue Systems and Conversational AI",
        "Ethical Implications of AI in Healthcare",
        "Neural Machine Translation (NMT)",
        "Time Series Forecasting with Machine Learning",
        "Multi-modal Learning and Fusion",
        "Named Entity Recognition (NER) in NLP",
        "Human Pose Estimation in Computer Vision",
        "Language Generation Models",
        "Cognitive Robotics and AI Integration",
        "Visual Question Answering (VQA) Systems",
        "Privacy and Security in AI Applications",
        "Graph Neural Networks (GNNs) for Structured Data",
        "Introduction to Python programming",
        "Python data types and variables",
        "Control flow and loops in Python",
        "Functions and modules in Python",
        "File handling in Python",
        "Object-oriented programming (OOP) in Python",
        "Error handling and exceptions in Python",
        "Python libraries for data analysis (e.g., Pandas, NumPy)",
        "Web scraping with Python (e.g., using BeautifulSoup)",
        "Creating GUI applications in Python (e.g., using Tkinter)",
        "History of Formula 1 racing",
        "Formula 1 car specifications and regulations",
        "Famous Formula 1 drivers and their achievements",
        "Formula 1 circuits around the world",
        "How Formula 1 teams operate and strategize",
        "Technological innovations in Formula 1",
        "Role of aerodynamics in Formula 1 cars",
        "Formula 1 race formats (qualifying, practice sessions, race day)",
        "Evolution of safety measures in Formula 1",
        "Economic impact of Formula 1 on host countries",
        "Formula 1 engine specifications and development",
        "Famous rivalries in Formula 1 history",
        "Formula 1 team dynamics and hierarchy",
        "How Formula 1 impacts automotive technology",
        "The role of tire management in Formula 1 races",
        "Key differences between Formula 1 and other racing series",
        "The influence of sponsors in Formula 1",
        "Formula 1 rules and regulations changes over the years",
        "Notable controversies in Formula 1",
        "The future of Formula 1 racing"
        ]
    print(len(queries))
    for query in queries:
        print(query)
        transcript_file = search_and_transcribe_videos(query)
        print(transcript_file)
        time.sleep(5)

        pdf_filename = create_pdf(transcript_file)
        time.sleep(10)

        add_documents_to_db(pdf_filename)

    return True

import tiktoken

def update_conversation_summary(summarized_conversation, new_interaction):
    """
    Update the summary of a conversation by appending a new interaction.
    
    Args:
        summarized_conversation (str): The current summarized conversation.
        new_interaction (dict): A dictionary containing 'question' and 'answer' keys.
    
    Returns:
        str: The updated summary of the conversation.
    """

    new_summary = f"{summarized_conversation}\n- Q: {new_interaction['question']}\n  A: {new_interaction['answer']}"
        
    return new_summary


def is_long_task(task, max_tokens=1000):
    """
    Determine if a given task exceeds the specified token limit.
    
    Args:
        task (str): The task to check.
        max_tokens (int): The maximum number of tokens allowed.
    
    Returns:
        bool: True if the task exceeds the token limit, False otherwise.
    """

    encoding = tiktoken.encoding_for_model(llm)
    num_tokens = len(encoding.encode(task))
    return num_tokens > max_tokens

def split_task(task):
    """
    Split a long task into smaller subtasks for easier processing.
    
    Args:
        task (str): The task to split.
    
    Returns:
        list of str: A list of subtasks.
    """

    prompt = f"""
    The following task needs to be split into smaller subtasks:
    
    {task}
    
    Please divide this task into 2-4 subtasks. Each subtask should be a complete, standalone task.
    Format your response as a Python list of strings, with each string being a subtask.
    """
    
    response = llm.invoke(prompt)
    subtasks = eval(response)
    return subtasks

def combine_results(results):
    """
    Combine the results from multiple subtasks into a single summary.
    
    Args:
        results (list of str): The results from subtasks.
    
    Returns:
        str: A concise summary of the combined results.
    """

    combined = "Combined results from subtasks:\n\n"
    for i, result in enumerate(results, 1):
        combined += f"Subtask {i} result:\n{result}\n\n"
    
    summary_prompt = f"""
    Please provide a concise summary of the following combined results:
    
    {combined}
    
    Summarize the key points and overall conclusion.
    """
    
    response = llm.invoke(summary_prompt)
    return response



def process_user_input(user_input):
    """
    Process user input by determining if it's a long task. If so, split it into subtasks,
    process each subtask, and combine the results. Otherwise, process the input directly.
    
    Args:
        user_input (str): The user's input to process.
    
    Returns:
        str: The result after processing the user input.
    """

    if is_long_task(user_input):
        subtasks = split_task(user_input)
        results = []
        for subtask in subtasks:
            result = run_agent(subtask)
            results.append(result)
        return combine_results(results)
    else:
        return run_agent(user_input)

# Uncomment the line below if you want to re-initialize the collection or initialize it with different topics
#initialize_collection()

def create_qa_chain():
    """
    Create a question-answering chain using a retriever and a language model.
    
    Returns:
        RetrievalQA: The question-answering chain instance.
    """

    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

def combine_summaries(summaries):
    """
    Combine multiple summaries into a single summary.
    
    Args:
        summaries (list of str): The list of summaries to combine.
    
    Returns:
        str: The combined summary.
    """

    combined_summary = " ".join(summaries)
    return combined_summary

def split_text(text, max_length=1500):
    """
    Split a long text into smaller chunks, ensuring chunks do not exceed the specified length.
    
    Args:
        text (str): The text to split.
        max_length (int): The maximum length of each chunk.
    
    Returns:
        list of str: A list of text chunks.
    """

    chunks = []
    while len(text) > max_length:
        chunk = text[:max_length]
        # Find the last complete sentence within the chunk
        last_period = chunk.rfind('. ')
        if last_period != -1:
            chunk = chunk[:last_period+1]
        chunks.append(chunk)
        text = text[len(chunk):].lstrip()
    if text:
        chunks.append(text)
    return chunks

def process_large_text(transcript_text):
    """
    Process a large text by splitting it into chunks, summarizing each chunk, 
    and then generating a final summary from the combined chunk summaries.
    
    Args:
        transcript_text (str): The large text to process.
    
    Returns:
        str: The final summary of the large text.
    """

    # Step 1: Split the cleaned text into manageable chunks
    chunks = split_text(transcript_text, max_length=1500)

    # Step 2: Generate summaries for each chunk
    chunk_summaries = [text_summarize(chunk) for chunk in chunks]

    # Step 3: Combine the chunk summaries
    combined_summary = combine_summaries(chunk_summaries)

    # Step 4: Generate the final summary from combined summaries
    final_summ = text_summarize(combined_summary)

    return final_summ

# Initialize memory with k=5, so the memory object will store the most recent 5 messages or interactions in the conversation
memory = ConversationBufferWindowMemory(k=5)

# Define agent tools
@tool
def search_kb(query):
    """
    Search the knowledge base for relevant documents based on a query and return a response.

    Args:
        query (str): The search query.

    Returns:
        str: The result from the QA chain based on the retrieved documents.
    """

    retriever = vector_db.as_retriever()
    docs = retriever.get_relevant_documents(query)
    summaries = "\n\n".join([doc.page_content for doc in docs])
    qa_chain = create_qa_chain()
    llm_response = qa_chain({"query": query})
    return llm_response["result"]

@tool
def process_video(url):
    """
    Processes a YouTube video by extracting its transcript, summarizing it, 
    and adding the transcript to the knowledge base.

    Args:
        url (str): The URL of the YouTube video to process.

    Returns:
        str: The summary of the video.
    """
#    video_id = url.split('v=')[-1]
#    transcript = YouTubeTranscriptApi.get_transcript(video_id)
#    transcript_text = ' '.join([t['text'] for t in transcript])

    audio_file = download_video_mp3(url)
    transcript_text = audio_to_text(audio_file)

    # Clean the transcript text
    cleaned_text = clean_text(transcript_text)
    if len(cleaned_text) > 15000:
        process_large_text(cleaned_text)
    
    # Generate a summary for the user
    summary = text_summarize(cleaned_text)
    
    print(f"Added {len(summary)} chunks from YouTube video {url} to the collection.")
    return summary

   
@tool
def new_search(query):
    """
    Perform a new search on YouTube, transcribe videos, create a PDF from the transcript, add documents to the database, and search the knowledge base.

    Args:
        query (str): The search query.

    Returns:
        str: The path to the created PDF file.
    """
    transcript = search_and_transcribe_videos(query)
    time.sleep(10)
    pdf_file = create_pdf(transcript)
    time.sleep(10)
    add_documents_to_db(pdf_file)
    time.sleep(5)
    search_kb(query)
    return pdf_file

@tool
def process_pdf(pdf):
    """
    Processes a PDF File by summarizing it, 
    and adding it to the knowledge base.

    Args:
        pdf (str): The path to the PDF file to process.

    Returns:
        str: The summary of the PDF.
    """

    loader = PyPDFLoader(pdf)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    
    return summary



# Define the agent tools
tools = [
    Tool(
        name="Search KB",
        func=search_kb,
        description="useful for when you need to answer questions about Machine Learning, Computer Vision and Natural Language Processing. The input to this tool should be a complete english sentence.",
    ),
    Tool(
        name="Search YouTube",
        func=new_search,
        description="useful for when the user asks you a question outside of Machine Learning, Computer Vision and Natural Language Processing. You use it to find new information about a topic not in the knowledge base. The input to this tool should be a complete english sentence.",
    ),
    Tool(
        name="Process Video",
        func=process_video,
        description="Useful for when the user wants to summarize or ask questions about a specific YouTube video. The input to this tool should be a YouTube URL.",
    ),
    Tool(
        name="Process PDF",
        func=process_pdf,
        description="Useful for when the user wants to summarize or ask questions about a specific PDF file. The input to this tool should be a PDF file path.",
    )
]



# Define the agent prompt
prompt_template_string  = """
You are an AI trained on Artificial Intelligence topics and Formula 1.


Answer the following questions as best you can, taking into account the context of the conversation.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action you should take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question


Example 1:
Question: What are dinosaurs?
Thought: I need to check the knowledge base for information on dinosaurs.
Action: Search Knowledge Base
Action Input: What are dinosaurs?
Observation: I don't have information on dinosaurs based on the provided context about machine learning and artificial intelligence.
Thought: I need to find new information about dinosaurs.
Action: Search YouTube
Action Input: Dinosaurs
Observation: Found relevant information and updated the knowledge base.
Thought: Now I can find information in the updated knowledge base.
Action: Search Knowledge Base
Action Input: What are dinosaurs?
Observation: [detailed information about dinosaurs]
Thought: I now know the final answer.
Final Answer: [final detailed answer about dinosaurs]

Example 2:
Question: Can you summarize this video? https://www.youtube.com/watch?v=dQw4w9WgXcQ
Thought: I need to extract the link to the video to get the summary.
Action: Process input to get link
Action Input: https://www.youtube.com/watch?v=dQw4w9WgXcQ
Observation: [summary of the video]
Thought: Now I can provide the summary of the video.
Final Answer: [summary of the video]

Example 3:
Question: Explain the content of this video https://www.youtube.com/watch?v=dQw4w9WgXcQ and how it relates to machine learning.
Thought: I need to extract the YouTube link from the input.
Action: Extract YouTube Link
Action Input: Explain the content of this video https://www.youtube.com/watch?v=dQw4w9WgXcQ and how it relates to machine learning.
Observation: Extracted YouTube link: https://www.youtube.com/watch?v=dQw4w9WgXcQ
Thought: I need to process the video to get the summary.
Action: Process Video
Action Input: https://www.youtube.com/watch?v=dQw4w9WgXcQ
Observation: [summary of the video]
Thought: Now I can relate the content to machine learning.
Final Answer: [explanation of how the video content relates to machine learning]

Example 4:
Question: Who are you?
Thought: I should explain that I'm a chatbot and how I can help.
Final Answer: I am a chatbot that can answer questions about machine learning and other related topics.

Example 5:
Question: What is your name?
Thought: I don't know.
Final Answer: I don't know the answer for that.

Question: {input}
{agent_scratchpad}"""

# Define the agent
prompt = PromptTemplate.from_template(prompt_template_string)


agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,handle_parsing_errors=True)



# Streamlit App Interface Design
def main():

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_summary' not in st.session_state:
        st.session_state.conversation_summary = ""

    # Function to clear chat history
    def clear_chat():
        st.session_state.messages = []

    st.title("AI Knowledge Base & Chat")

    # Fixed description at the top
    st.markdown("""
    **Welcome to the AI Knowledge Base & Chat App!** 🤖💬

    This interactive application leverages a sophisticated AI model to provide in-depth information and insights across a diverse range of topics. Here’s what you can explore:

    - **Artificial Intelligence and Machine Learning** 🌐
    - **Computer Vision** 👁️
    - **Python Programming** 🐍
    - **Formula 1 Racing** 🏎️

    With its extensive training on these topics, the AI is well-equipped to provide accurate, detailed, and relevant answers to your questions. Enjoy exploring a world of knowledge and get instant responses to your queries! 🎓✨
    In addition to answering your questions, you can:

    Upload a PDF File 📄: Submit a PDF document to have it automatically summarized, giving you a concise overview of its contents without having to read through the entire file.

    Provide a YouTube URL 🎥: Enter a link to a YouTube video to receive a summary of its key points, allowing you to grasp the main ideas quickly.
    """)
    
    # Layout for additional inputs and chat
    with st.sidebar:
        st.header("Additional Inputs")

        youtube_url = st.text_input("Enter YouTube URL:")
        if st.button("Process YouTube Video"):
            with st.spinner("Processing YouTube video..."):
                summary = process_video(youtube_url)
                st.write(summary)
                st.session_state.messages.append({"role": "assistant", "content": f"I've processed the YouTube video. Here's a summary:\n\n{summary}"})
                st.rerun()
                
        uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                texts = extract_text_from_pdf(uploaded_pdf)
                pdf_summary = text_summarize(texts)
                st.write(pdf_summary)
                st.session_state.messages.append({"role": "assistant", "content": f"PDF processed and added to knowledge base. Here's a summary:\n\n{pdf_summary}"})
                st.rerun()

    st.header("Chat")

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            with st.chat_message(role):
                st.markdown(content)
        else:
            with st.chat_message(role):
                st.markdown(content)

    user_input = st.chat_input("Ask a question")
        
        # Button to clear chat
    if st.button('Clear Chat'):
        clear_chat()

    if user_input:
    # Display user message
        with st.chat_message("user"):
            st.write(user_input)

            # Get AI response
        with st.chat_message("assistant"):
            response = agent_executor.invoke({"input": user_input})
            st.write(response['output'])
            st.session_state.messages.append({"role": "assistant", "content": response['output']})

if __name__ == "__main__":
    main()
