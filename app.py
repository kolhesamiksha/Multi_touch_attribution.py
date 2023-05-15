import streamlit as st 
from streamlit_option_menu import option_menu
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from minio import Minio
from PIL import Image, ImageOps
from io import BytesIO
import plotly.express as px
import hydralit_components as hc
import utils as ut
import os
import requests
import streamlit.components.v1 as components
import uuid
from katonic.pipeline.pipeline import dsl, create_component_from_func, compiler, Client
import kfp 
import os
os.system("pip install pandas pmdarima statsmodels minio")
from minio import Minio
import statsmodels.api as sm

ACCESS_KEY = "DNPD2SAYLBELJ423HCNU"
SECRET_KEY = "zF7F6W93HS8vt+JKen4U17+zhcHiwH47AMuO3ap0"
PUBLIC_BUCKET = "shared-storage"

client = Minio(
        endpoint="minio-server.default.svc.cluster.local:9000",
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False,
    )

usr_journey_path = "/multi_touch_attribution/usr_journey_attribution_data.csv"
usr_attribution_path = "/multi_touch_attribution/usr_attribution_data.csv"
transition_prob_matrix = "/multi_touch_attribution/transition_matrix.csv"
channel_performance_across_methods = "/multi_touch_attribution/channel_performance_across_methods.csv"
markov_chain_rm_data = "/multi_touch_attribution/markov_chain_attribution_update.csv"

client.fget_object(
        PUBLIC_BUCKET,
        usr_journey_path,
        "data/usr_journey_attribution_data.csv",
    )

client.fget_object(
        PUBLIC_BUCKET,
        usr_attribution_path,
        "data/usr_attribution_data.csv",
    )

client.fget_object(
        PUBLIC_BUCKET,
        transition_prob_matrix,
        "data/transition_matrix.csv",
    )

client.fget_object(
        PUBLIC_BUCKET,
        channel_performance_across_methods,
        "data/channel_performance_across_methods.csv",
    )

client.fget_object(
        PUBLIC_BUCKET,
        markov_chain_rm_data,
        "data/markov_chain_rm_data.csv",
    )

routes = os.environ["ROUTE"]

response = requests.get(url='https://katonic.ai/favicon.ico')
im = Image.open(BytesIO(response.content))

st.set_page_config(
    page_title='Multi-Touch-Attribution-markov-chain', 
    page_icon = im, 
    layout = 'wide', 
    initial_sidebar_state = 'auto'
)
  
menu_data = [
        {'icon': "🚀", 'label':"Load_data-katonic-connectors"},
        {'icon':"🔃",'label':"data-preparation",'submenu':[{'icon': "fa fa-paperclip", 'label':"User_Journey"},{'icon': "fa fa-paperclip", 'label':"Heuristic_attribution"}]},
        {'icon': "💻", 'label':"markov-chain-model",'submenu':[{'icon': "fa fa-paperclip", 'label':"Markov-chain-attribution"},{'icon': "fa fa-paperclip", 'label':"heuristic v/s data-driven"}]},
        {'icon': "far fa-chart-bar", 'label':"Spend_optimisation",'submenu':[{'icon': "fa fa-paperclip", 'label':"use-case"},{'icon': "fa fa-paperclip", 'label':"View-performance"},{'icon': "fa fa-paperclip", 'label':"Budget-allocation"}]},
        {'icon': "➰", 'label':"kubeflow-pipeline"},
]

# we can override any part of the primary colors of the menu
#over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home',override_theme=over_theme)

if menu_id=="Home":

    st.title("Multi-Touch-Attribution")

    st.write("""
    Behind the growth of every consumer-facing product is the **acquisition and retention of an engaged user base**. 
    When it comes to acquisition, the goal is to attract high quality users as cost effectively as possible. With marketing dollars dispersed across a wide array of campaigns, channels, and creatives, however, measuring effectiveness is a challenge. 
    In other words, it's difficult to know how to assign credit where credit is due. """)

    st.write("""Enter multi-touch attribution. With multi-touch attribution, credit can be assigned in a variety of ways, but at a high-level, it's typically done using one of two methods: **heuristic or data-driven**.
    """)

    st.write("""
    #### Heuristic based approach
    """)

    st.write("""
    heuristic methods are rule-based and consist of both single-touch and multi-touch approaches:
    1. **Single-touch**: Single-touch methods, such as first-touch and last-touch, assign credit to the first channel, or the last channel, associated with a conversion.

    2. **Multi-touch**: Multi-touch methods, such as linear and time-decay, assign credit to multiple channels associated with a conversion.
    """)

    st.write("""
    #### Data-driven approach
    data-driven methods determine assignment using probabilities and statistics. Examples of data-driven methods include **Markov Chains and SHAP**.  
    """)

    st.write("""
    #### 
    
    """)

if menu_id=="kubeflow-pipeline":
    st.title("kubeflow-pipeline")
    if st.button("Trigger a Pipeline Instance"):
        @create_component_from_func
        def data_loading_from_aws():
            print("data_loading")

        @create_component_from_func
        def data_preparation(draft_path:str):
            print("data_preparation")

        @create_component_from_func
        def markov_chain_model(draft_path:str):
            print("markov_chain_model")

        @create_component_from_func
        def spend_optimisation(draft_path:str):
            print("X")

        @dsl.pipeline(
            name='Markov-multi-touch-attribution',
            description='end-to-end Markov-multi-touch-attribution use-case'
        )

        def markov_multi_touch_chain_attribution_model():
            import os

            data_loading_from_aws_task = data_loading_from_aws()
            data_preparation_task = data_preparation(data_loading_from_aws_task)
            markov_chain_model_task = markov_chain_model(data_preparation_task)
            spend_optimisation(markov_chain_model_task)
        
        pipeline_func = markov_multi_touch_chain_attribution_model
        pipeline_filename = pipeline_func.__name__ + f'{uuid.uuid1()}.pipeline.yaml'
        compiler.Compiler().compile(pipeline_func, pipeline_filename)
        client = Client()
        experiment = client.create_experiment('Markov-multi-touch-attribution')
        run_name = f"{pipeline_func.__name__}{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
        client.upload_pipeline(pipeline_filename)
        run_result = client.run_pipeline(experiment.id, run_name, pipeline_filename)
        st.markdown('<p style="color:Green;">Brew yourself a cup of coffee, while we produce your result...</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:Green;">Please, don\'t interrupt this tab while pipeline is running.</p>', unsafe_allow_html=True)
        pipeline_link = f'https://devenv.katonic.ai/pipeline/#/runs/details/{run_result.id}'
        st.write("[Goto Pipeline](%s)" % pipeline_link)
        res = client.wait_for_run_completion(run_result.id, 36000)
        if res.run.status == 'Failed':
            st.error('Pipeline Failed.')
        if res.run.status == "Succeeded":
            st.error("Pipeline Successfully completed.")

if menu_id=="Load_data-katonic-connectors":
    st.title("Load_data using Katonic Connectors")
    st.write("##### Loaded the data residing inside **AWS S3** bucket to the katonic **file-manager** using katonic connectors")
    st.write("""
        just by providing Source as IAM access key, secrete key, bucket name and object name 

        Destination as a katonic_file manager access and secrete key, bucket_name:private/public
    """)

    st.write("""
        Loaded data from AWS S3 to Katonic-filemanager
    """)

    img1 = Image.open('image/katonic_connectors.png')
    st.image(img1)

if menu_id=="data-preparation":
    st.write("xsdea")

if menu_id=="User_Journey":
    st.title("Users Journey🚀")

    st.write("""
    ###### A user journey is defined as a series of steps that represent how you want users to interact with your app. It involves the analysis of how users are interacting with the app to identify the weakest points in the path to conversion.
    ###### Mapping the user journey from the time of download and their first session is important for your app’s growth 
    """)

    st.write("""

    ##### Points can be drawn up by understanding users journey map: 

    - Where the audience interacts with the product. This can be a company website, ads on third-party resources, social media, etc. 
    - How various audience segments interact with your app. 
    - What stages does the user go through before buying and what goals they have.
    
    """)
    img1 = Image.open('image/user_journey.jpg')
    st.image(img1,width=500)

    usr_journ_data = pd.read_csv("data/usr_journey_attribution_data.csv")
    if st.button("User_journey_data"):
        st.dataframe(usr_journ_data)

if menu_id=="Heuristic_attribution":
    st.title("Heuristic approach for user attribution")

    st.write("""
        ###### Heuristic means a practical method of achieving a goal, that is not necessarily optimal, but is good enough for now. Heuristic methods are used when figuring out the optimal way to do something is too costly,
        ###### The heuristic attribution models are all based on where in the customer journey you are.. 
    """)
    st.write("---")
    st.write("""
    There are two main types of methods (or models) used to divide the revenue across channels.""")

    st.write("""
    1. **First Touch Attribution**: This model assigns 100% of the opportunity value to the first (oldest) interaction with the customer.
    First touch is helpful when you want to know which channels are most likely to bring new leads. 
    """)

    st.write("""
    2. **Last Touch Attribution**: This model assigns 100% of the opportunity value to the most recent interaction with the customer. Basically, the last campaign that the prospect interacted with gets all the credit.
    Use last touch to figure out which campaigns / channels are your best deal closers. 
    """)

    usr_attr_data = pd.read_csv("data/usr_attribution_data.csv")
    if st.button("Heuristic_attribution_analysis"):
        st.dataframe(usr_attr_data)

        cnt_plot = sns.catplot(x='channel',y='attribution_percent',hue='attribution_model',data=usr_attr_data, kind='bar', aspect=2).set_xticklabels(rotation=15)
        st.pyplot(cnt_plot)

if menu_id=="Dashboard-analysis":
    st.write("INNJXFSFV")

if menu_id=="Markov-chain-attribution":
    st.title("Markov-chain-attribution modelling")

    st.write("""##### Markov chaining is the process/illustration of predicting the future event based on some previous conditions/event.""")
    st.write("---")
    st.write("""
    Three steps to calculate attribution using Markov chains:
    1. **Transition probability matrix**: A transition probability matrix is a matrix that contains the probabilities associated with moving from one state to another state. This is calculated using the data from all available customer journeys.

    2. **Total conversion probability**: on average, the likelihood that a given user will experience a conversion event.

    3. **Removal effect per channel**: difference between the total conversion probability (with all channels) and the conversion probability when the conversion is set to 0%.
    """)

    trans_data = pd.read_csv('data/transition_matrix.csv')
    markov_chain_data = pd.read_csv("data/markov_chain_rm_data.csv")
    if st.button("Transition_Matrix"):
        st.write("##### Transition Probability Matrix")
        st.dataframe(trans_data)
        st.write("---")
        st.write("##### Total Conversion Probability")
        st.success("0.03760307804217135")
        st.write("---")
        st.write("##### Removal Effct per chnnel probability")
        st.dataframe(markov_chain_data)

if menu_id=="heuristic v/s data-driven":
    st.title("Heuristic v/s Data-driven")
    st.write("""
    **the multi-touch models are more accurate**, since you’re not betting all your money on a single channel. 
    
    It’s only fair to say all the interactions your lead had with your company contributed to some extent to the conversion, 
    
    so to credit only the first one, the last one, or some other one in the middle may seem simplistic and reductionist.
    """)

    new_df = pd.read_csv('data/channel_performance_across_methods.csv')
    if st.button("channel_performance"):
        st.dataframe(new_df) 

    hr_plot = sns.catplot(x='channel',y='attribution_percent',hue='attribution_model',data=new_df, kind='bar', aspect=2).set_xticklabels(rotation=15) 
    st.pyplot(hr_plot)

if menu_id=="use-case":
    st.write("in this use case")

if menu_id=="View-performance":
    st.write("sxzcdafwr")

if menu_id=="Budget-allocation":
    st.write("sfligrkldc")