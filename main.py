""" DELTAI FELLOWSHIP : PROJECT 1 - AI CHATBOT APPLICATION """

#######################################################################################################

# """ NECESSARY IMPORTS """

import streamlit as st # alias 
import openai
import pandas as pd

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS # Facebook library for similarity search of text
from langchain.llms import OpenAI #langchain.OpenAI is just a wrapper & openAI not belong to langchain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain_experimental.agents import create_pandas_dataframe_agent

from PyPDF2 import PdfReader
import time
from dotenv import load_dotenv
import os

#####################################################################################################

# To view app on browser : streamlit run main.py --browser.serverAddress localhost
# First click the play button and wait till ' streamlit run c:/.../....> ' appears
# Then enter command given above ( everything starting from streamlit to localhost )


# Load environment variable and assign as API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#####################################################################################################

# """ MAIN FUNCTION FOR CREATING NAVBAR AND LOADING DIFFERENT PAGES """

def main():

    # All design and structural elements occupy wider area 
    st.set_page_config('wide')

    # Create sidebar - On button click call function corresponding to that page
    st.sidebar.title("Navigation Bar")
    pages = {
        "Home": homepage,
        "My Chatbot": chatbot_page,
        "Article Generator": article_generator,
        "ChatCSV": chat_csv,
        "ChatPDF": chat_pdf,
        "DALL-E" : image_generator
    }

    selected_page = st.sidebar.button("Home", key="home",use_container_width=True,type='primary')
    if selected_page:
        # URL in the browser's address bar will be updated to include the query parameter 'page=home'
        st.experimental_set_query_params(page="home")
    selected_page = st.sidebar.button("Chatbot", key="chatbot",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="chatbot")
    selected_page = st.sidebar.button("Article Generator", key="seo_article",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="seo_article",)
    selected_page = st.sidebar.button("ChatCSV", key="chatcsv",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="chatcsv")
    selected_page = st.sidebar.button("ChatPDF", key="chatpdf",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="chatpdf")
    selected_page = st.sidebar.button("DALL-E", key="dall_e",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="dall_e")

    # Get the page name from the URL, default to "home"
    page_name = st.experimental_get_query_params().get('page', ['home'])[0]

    # Call the corresponding page based on the selected page_name
    if page_name == "home":
        homepage()
    elif page_name == "chatbot":
        chatbot_page()
    elif page_name == "seo_article":
        article_generator()
    elif page_name == "chatcsv":
        chat_csv()
    elif page_name == "chatpdf":
        chat_pdf()
    elif page_name == "dall_e":
        image_generator()

##################################################################################################

# """ HOMEPAGE WITH DESCRIPTIONS OF VARIOUS TOPICS RELATED TO LLM, ChatGPT, ETC. """

def homepage():

    # Custom CSS for homepage spefically for content containers
    st.markdown(
        """
        <style>
        .homepage-subheading {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .gpt-example-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 2rem;
        }
        .gpt-example-box {
            width: 45%;
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            background-size: 200% 100%;
            animation: gradientAnimation 3s linear infinite;
        }
        .gpt-example-box:nth-child(1) {
            background-image: linear-gradient(45deg, #1E90FF 0%, #4682B4 100%);
        }
        .gpt-example-box:nth-child(2) {
            background-image: linear-gradient(45deg, #32CD32 0%, #228B22 100%);
        }
        .gpt-example-box:nth-child(3) {
            background-image: linear-gradient(45deg, #9370DB 0%, #6A5ACD 100%);
        }
        .gpt-example-box:nth-child(4) {
            background-image: linear-gradient(45deg, #FFA500 0%, #FF8C00 100%);
        }
        .gpt-example-box p {
            font-size: 16px;
            margin: 0;
            color: white;
        }
        .gpt-example-box h3 {
            font-size: 20px;
            margin-bottom: 0.5rem;
            color: white;
        }

        @keyframes gradientAnimation {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Page title
    st.title("Stark.AI - Vedanth Aggarwal") 

    # Section 1 : LLM Versions of ChatGPT
    st.markdown('<div class="homepage-subheading">Different LLM Versions of ChatGPT</div>', unsafe_allow_html=True)
    st.write("Here you can find a descriptio of Language Model (LLM) models that have been used by ChatGPT. Each model comes with unique capabilities and features.")

    # Content boxes with version name and description
    with st.container():
        st.markdown(
            """
            <div class="gpt-example-container">
                <div class="gpt-example-box">
                    <h3>GPT-3</h3>
                    <p>This was the first version to be used in the creation of ChatGPT,it was trained on large amounts of text
                            data, allowing it to predict the next word in a given sequence. It excelled at tasks
                              such as answering questions and generating text, and was especially good at the
                                NLP tasks that are central to chatbot functionality
                    </p>
                </div>
                <div class="gpt-example-box">
                    <h3>GPT-3.5</h3>
                    <p>When ChatGPT was released for public use on November 30, 2022,
                      it was running on a model fine-tuned from the GPT-3.5 series,
                        an improved model from the original GPT-3. The GPT-3.5 version could engage with a range
                          of topics, including programming, TV scripts, and scientific concepts</p>
                </div>
                <div class="gpt-example-box">
                    <h3>GPT-3.5-Turbo</h3>
                    <p>This version had 175 billion parameters, significantly more than GPT-3.5.With this
                      additional complexity, GPT-3.5 Turbo could perform tasks such as writing and debugging
                        computer programs, composing music, generating business ideas,
                          and emulating a Linux system, among other things</p>
                </div>
                <div class="gpt-example-box">
                    <h3>GPT-4</h3>
                    <p>This version has over a trillion paramaters and is a multimodel that accepts text and image
                    input. It has a much longer memory (up to 64,000 words), improved multilingual capabilities,
                      more control over responses, and a limited search capacity that allows it to pull text
                        from web pages when a URL is shared in the prompt. It also introduced the ability to work with plugins, allowing third-party developers to make ChatGPT-4 "smarter"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section 2 : Special parameters in OpenAI API requests
    st.markdown('<div class="homepage-subheading">Key Terminologies in OpenAI API requests</div>', unsafe_allow_html=True)
    st.write("These parameters in openAI requests have a significant influence on the output generated, try experimenting with them in the openAI playground")

    with st.container():
        st.markdown(
            """
            <div class="gpt-example-container">
                <div class="gpt-example-box">
                    <h3>Temperature</h3>
                    <p>Temperature controls the randomness of the model's output. Higher values like 0.8 make the output more diverse, while lower values like 0.2 make it more focused and deterministic.</p>
                </div>
                <div class="gpt-example-box">
                    <h3>Top-p</h3>
                    <p>Top-p (nucleus) sampling truncates the model's output to the most probable tokens that cumulatively exceed a given probability threshold (e.g., 0.9). This prevents the model from producing overly rare or nonsensical tokens.</p>
                </div>
                <div class="gpt-example-box">
                    <h3>Presence Penalty</h3>
                    <p>Presence Penalty is used to discourage the model from generating certain tokens in its output. By adding a presence penalty, you can avoid getting specific types of responses.</p>
                </div>
                <div class="gpt-example-box">
                    <h3>Frequency Penalty</h3>
                    <p>Frequency Penalty controls the amount of repetition in the model's responses. Higher values like 2.0 reduce repetitive behavior, while lower values like 0.2 allow for more repetition.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Section 3 : Other LLM Models
    st.markdown('<div class="homepage-subheading">Large Language Models by Other Entities</div>', unsafe_allow_html=True)
    st.write("These are descriptions of a few other LLM models developed by other companies and institutes.")
    
    with st.container():
        st.markdown(
            """
            <div class="gpt-example-container">
                <div class="gpt-example-box">
                    <h3>PaLM 2 (Bison-001) by Google:</h3>
                    <p>This model is part of Google's PaLM 2 series, and it focuses
                      on commonsense reasoning, formal logic, mathematics, and advanced coding in over 20 languages.
                        Trained on 540 billion parameters and has a maximum
                          context length of 4096 tokens. It's also a multilingual model that can understand idioms,
                            riddles, and nuanced texts from different languages</p>
                </div>
                <div class="gpt-example-box">
                    <h3>Falcon by the Technology Innovation Institute (TII), UAE</h3>
                    <p>This open-source LLM that has been trained on 40 billion parameters
                      (Falcon-40B-Instruct model). It has been primarily trained in English, German, Spanish,
                        and French but can also work in several other languages like Italian, Portuguese,etc.
                          Falcon's open-source nature allows it to be used for commercial purposes without restrictions</p>
                </div>
                <div class="gpt-example-box">
                    <h3>RoBERTa by Facebook</h3>
                    <p>RoBERTa, developed by Facebook, is a variant of BERT that uses a different training approach.
                      It's trained on a larger amount of data, uses larger batches and longer sequences during
                        training, and removes the next-sentence prediction task that BERT uses.
                          These changes result in a model that outperforms BERT on several benchmark tasks</p>
                </div>
                <div class="gpt-example-box">
                    <h3>Vicuna 33B by LMSYS</h3>
                    <p>Vicuna is an open-source LLM derived from LLaMA.
                      It is trained on 33 billion parameters and has been fine-tuned using supervised instruction.
                        Despite its smaller size compared to some proprietary models,
                          Vicuna has shown remarkable performance</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

#########################################################################################################

# """ STAR FEATURE : CHATBOT APPLICATION USING OPENAI API"""

def chatbot_page():

    st.title("Welcome To My Chatbot")
    col1,col2,col3 = st.columns(3) # Create 3 seperate columns for structure and alignment
    options = ["Default", "Funky Freak", "Sassy Sam",'Whisker Wise']

    with col2:
        # Default + Unique personalities for user to experiment with
        personality = st.selectbox("Personality Choices", options,label_visibility='collapsed',index=options.index('Default'))
    with col3:
        submit_button1 = st.button('Select Personality',type='primary',use_container_width=True)
    with col1:
        clear_button = st.button('Clear Chat',use_container_width=True,type='primary')

    # Assign AI Chatbot role/personality
    content = "" 
    if submit_button1:
        if personality=='Funky Freak':
            content ="""->->->You are now a funky 
                        freak personality, make jokes, dark humour, add cringe statements. Have a skaterboard vibe
                        add weird crazy comments, freak out with panic attacks and keep making hilarious joked, puns and talk about memes
                        Reply with a greeting to the user embodying this personality"""
        elif personality=='Sassy Sam':
            content = """->->->You are now an extremely sassy personality, boast, make sassy remarks, offer unwanted advice.
                        Keep focusing on yourself, praise yourself, make excuses and be very judgemental. Make comments,
                        and just be extremely SASSSYYY!! Reply with a greeting to the user embodying this personality"""
        elif personality=='Whisker Wise':
            content = """->->->You are now an extremely sarcastic yet wise personality. Be extremeley philosophical, keep branching out
                        into conversations about ethical dilemnas and the purpose of life. Warn the user of the future and their role in life. Be sarcastic yet 
                        prophesize about the world and give advice to the user. Reply with a greeting to the user embodying this personality"""
        else:
            content = """->->->You are an AI chatbot and your goal is to answer user queries. You have a neutral personality.
                        """
        temperature = 0.7
        st.session_state.messages.append({"role": "user",'content':content})

        # Add a standard assistant reply so openAI knows in message history this is the personality
        st.session_state.messages.append({"role": "assistant",
                                           "content": '->->->I have undertood and will answer all upcoming messages through this personality specifically and will reply embodying the personality described by you'})
    
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    
    colx,coly = st.columns(2) 
    with colx:
        # Model version ( Legacy + other models are deprecated/inaccessible )
        st.session_state['openai_model'] = st.selectbox("",['gpt-3.5-turbo','gpt-3.5-turbo-16k'],label_visibility='collapsed',index=0)
    with coly:
        # Temperature affects how random or standard the responses are 
        # Ex : The cat sits on the ____ (0.5 - mat, 1.7 - windowsill)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.7,format="Temp : %f", label_visibility='collapsed')
    

    if "messages" not in st.session_state:
        st.session_state.messages = [] # Create message history as empty list 
    
    if clear_button:
        st.session_state.messages = [] # Clear history
        st.session_state.messages.append({"role": "user", "content": '->->->You role now is to simply be an extremely helpful AI chatbot assistant and answer user queries. Keep no personality and be neutral'})
        st.session_state.messages.append({"role": "assistant",
                                           "content": '->->->I have undertood and will answer all upcoming messages through this personality specifically and will reply embodying the personality described by you'})

    # If messages pertain to the personality selection, don't display - better user experience
    for message in st.session_state.messages:
        if not message['content'].startswith('->->->'): # ->->-> special sequences to identify personality message
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # If user prompt is not empty
    if prompt:= st.chat_input('What is up?'):

        # Add and display user prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # OpenAI API request with streaming functionality
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],

                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ], temperature = temperature,
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

#########################################################################################################

# """ DALL-E API BASED IMAGE GENERATION ACCORDING TO USER PROMPT """

def image_generator():

    # Function that sends API request using user's prompt and returns image generated
    def generate_dall_e_image(prompt):
        try:
            response = openai.Image.create(
                model="image-alpha-001",  # DALL-E model
                prompt=prompt,
                n=1,  # Number of images to generate
            )
            image_url = response['data'][0]['url']
            return image_url
        
        except Exception as e:
            st.error(f"Error generating the image: {e}")
            return None
    
    # Working graphic interface
    st.title("Fun with DALL-E 💡")
    st.info("Enter the prompt below : ")
    prompt = st.text_area("Enter a prompt to generate an image:", "Hyper futuristic mars colony",label_visibility='collapsed')

    if st.button("Generate Image",type='primary'): # When button is clicked 
        if prompt.strip() == "":
            st.warning("Please enter a prompt to generate an image.") # Warning message if prompt empty
        else:
            with st.spinner("Generating image..."):
                image_url = generate_dall_e_image(prompt)
                if image_url:
                    st.image(image_url, caption="Generated Image", width=500) # Display image

####################################################################################################################

# """ ARTICLE/PARAGRAPH GENERATOR """


def article_generator():

    # Request OpenAI for response based on options/criteria chosen by user
    def generate_article(keyword, writing_style, word_count,article_type):
    
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "user", "content":f""" Write an {article_type} about ({keyword}) in a 
                    {writing_style} writing style with the length not exceeding {word_count} words"""}
                ]
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        return result

    # Working graphic interface
    st.title("Article Writer with ChatGPT 😲")

    # Options available to user for generating specific article/post
    keyword = st.text_input("Enter keywords:")
    writing_style = st.selectbox("Select writing style:", ["Casual","Informative", "Witty","Catchy",'Academic'])
    article_type = writing_style = st.selectbox("Select text type:", ["Essay", "Blog Article", "LinkedIn Post","Intagramm/Social Media post",'Summary Paragraph','Top 10 list'])
    
    col1,col2 = st.columns([0.8,0.2]) # Adjust width percentage for each column
    with col1:
        # Word count - Special Note : ChatGPT not great at following the word limit
        # Could replace it with selectbox and options short, medium, long
        word_count = st.slider("Words", min_value=50, max_value=1000, step=50, value=500,format="%d words", label_visibility='collapsed')
    with col2:
        submit_button = st.button("Generate Article",use_container_width=True,type='primary')

    if submit_button:
        # Simulate progress bar animation
        progress_bar = st.progress(0)
        for i in range(51):
            time.sleep(0.05)  
            progress_bar.progress(i)
        article = generate_article(keyword, writing_style, word_count,article_type) # Generate article
        for i in range(51,101):
            time.sleep(0.05)  
            progress_bar.progress(i)  

        st.info("Process completed!")
        st.write(article)

        # Download file as text - Add additional functionality
        st.download_button(
            label="Download",
            data=article,
            file_name='Article.txt',
            mime='text/txt',
        )

#################################################################################################

# """ CHATBOT THAT CAN INTERPRET CSV FILES AND ANSWER USER QUERIES ( DATA ANALYSIS ) """

def chat_csv():

    st.title('CHATCSV powered by LLM! 👾 ')
    st.info('Upload CSV file below : ')
    input_csv = st.file_uploader("Upload your CSV file",type=['csv'],label_visibility='collapsed') # File upload
    if input_csv is not None:
        st.info("CSV uploaded successfully!")
        data = pd.read_csv(input_csv)

        agent = create_pandas_dataframe_agent(OpenAI(temperature=0),data,verbose=True) 
        st.dataframe(data,use_container_width=True) # Visual dataframe with rolws and columns

        st.info('Enter your query..')
        input_text = st.text_area('Enter your query..',label_visibility='collapsed')
        if input_text != None:
            if st.button('Chat with CSV'):
                result = agent.run(input_text) # Thinking process shown in terminal
                st.success(result)
        else:
            st.warning('Error : No input query given')

#########################################################################################################

# """ CHATBOT THAT CAN READS PDFs USING PYPDF2, LANGCHAIN AND ANSWERS USER QUERIES """

def chat_pdf():

    st.title('Chat With Your PDF 🧑‍🚀')
    st.info('Upload your PdF file below : ')
    pdf = st.file_uploader("Upload your PDF file",type=['pdf'],label_visibility='collapsed')

    # Extract all text from PDF 
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks - ChatGPT cannot process extremely large message history from PDF
        # Have a overlap to ensure context in paragraphs that start from middle of sentences 
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        #Create embeddings ( Vector representations of text ) and store all chunks collection in 1 base
        embeddings = OpenAIEmbeddings()
        info_base = FAISS.from_texts(chunks,embeddings)

        user_question = st.text_input('Ask Question : ')
        if user_question:
            # RETRIEVAL AUGMENTED GENERATION
            # Find relevant chunks as per langchain model using similarity search of vectors
            docs = info_base.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm,chain_type="refine") # Handle question answering tasks 
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=user_question)
                st.info(f"Completion tokens : {cb.completion_tokens}") # Tokens used
            st.write(response)

##########################################################################################################

# Only if application is run directly ( not imported ), run the code
if __name__ == "__main__":
    main()