import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os
import requests
import json


# setup persentation page number
if 'steps' not in st.session_state:
    st.session_state.steps = -1
    st.session_state.valid = False

# Next button
_,_,_,_,_,_,_, col10 = st.columns(8)
with col10:
    if st.session_state.steps != 6:
        if st.button("Next"):
            st.session_state.steps += 1
            st.session_state.valid = False
st.markdown("""##""")

# the side bar 
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.steps != -1:
            st.image("images/logo.png")
    with col2:
        if st.session_state.steps != -1:
            st.title("Text Classification Pipeline")
    if st.session_state.steps == 1 :  # pipeline step 1 
        st.image("images/1.png")
    elif st.session_state.steps == 2: # pipeline step 2
        st.image("images/2.png")
    elif st.session_state.steps == 3: # pipeline step 3
        st.image("images/3.png")
    elif st.session_state.steps == 4: # pipeline step 4
        st.image("images/4.png")
    elif st.session_state.steps == 5: # pipeline step 5 
        st.image("images/5.png")
    elif st.session_state.steps == 6: # pipeline step 6
        st.image("images/6.png")

# 1st slide
if st.session_state.steps == -1:
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/logo.png")
    with col2:
        st.title("Text Classification Pipeline")

# 2nd slide
if st.session_state.steps == 1:
    t = r"""
    <p style="font-family:sans-serif; color:Black; font-size: 80px;"> Let's Classify </p>
    <p style="text-align: center; font-family:sans-serif; color:Green; font-size: 100px;"> A Resturent Reviews</p>
    <p style="text-align: right; font-family:sans-serif; color:Black; font-size: 80px;"> In 5 minutes </p>
"""
    st.markdown(t, unsafe_allow_html=True)

# 3rd slide 
elif st.session_state.steps == 0:
    st.markdown("""##""")

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.image("images/accurecy.png")
        st.subheader("Accuracy")
    with col3:
        st.image("images/implement.png")
        st.subheader("In-house Implementation")
    with col6:
        st.image("images/label.png")
        st.subheader("Labeling")

# 4th slide (which is step 1 in the pipeline)
elif st.session_state.steps == 2:
    st.header('Reviews Data')
    st.markdown("""##""")

    col1, col2, col3, col4, col5= st.columns(5)
    with col1:
        st.image("images/google.png")
        st.subheader("Source")
    with col3:
        st.image("images/rows.png")
        st.subheader("979 rows")
    with col5:
        st.image("images/lang_en.png")
        st.subheader("English Language")

    st.divider()

    # show the code 
    code = '''
    reviews_df.head(5)
    '''
    st.code(code, language='python')

    # run button
    if st.button(":running:"):
        file_path = 'data/Restaurant_Reviews.xlsx'
        data = pd.read_excel(file_path)
        st.dataframe(data[['Review']].head(5))

        col1, col2, col3= st.columns(3)
        with col2:
            st.image("images/prepare.png")
        st.dataframe(data[['Review']].head(5))

        st.divider()


        col1, col2 = st.columns(2)
        with col1:
            st.image("images/good.png")
            st.write("The pipeline requires minimal preprocessing.")
        with col2:
            st.image("images/bad.png")
            st.write("Challenge in splitting inputs that fall into several categories.")

# 5th slide (which is step 2 in the pipeline)
elif st.session_state.steps == 3:
    st.header('Discover what our reviews talk about')
    st.markdown("""##""")

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/bertopic.png")
    with col2:
        st.image("images/bertopic2.png")

    st.divider()

    code = '''
    from bertopic import BERTopic

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(reviews_df)

    # print extracted topics
    topic_model.get_topic_info()
    '''
    st.code(code, language='python')

    # run button
    if st.button(":running:"):
        st.session_state.valid = True
    if st.session_state.valid:
        topics = pd.read_excel("data/topics.xlsx")
        st.dataframe(topics[['Topic', 'Count', 'Name', 'Representation']])

        # show vis
        HtmlFile = open("../Fewshot-text-classification/identify_categories/iframe_figures/figure_8.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height=1000, width=800)

        # llm
        st.image("images/llm.png")
        code = '''
        prompt = f"""<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
        <|user|>
        I have a topic that contains the following documents:
        [REVIEWS]

        The topic is described by the following keywords: '[REVIEWS-KEYWORDS]'.

        Based on the information about the topic above, please create a short label of this topic. 
        Make sure you to only return the label and nothing more.</s>
        <|assistant|>"""
        '''
        st.code(code, language='python')

        # run button
        if st.button("run"):
            st.dataframe(topics[['Name','LLM_topic']])

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.image("images/good.png")
                st.write("It offers numerous impressive visualizations.")
            with col2:
                st.image("images/bad.png")
                st.write("Inconsistent executions with no clear criteria for selection.")

# 5th slide (which is step 3 in the pipeline)
elif st.session_state.steps == 4:
    st.header('Get some data labeled for model training')
    st.markdown("""##""")

    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader("3 Categories")
        code = '''
        target_topics = {
            0: 'Service variability slow vs excellent',
            2: 'Place feedback positive vs negative',
            7: 'Food quality- terrible, gross, dry, sucked, fried',
            }
        '''
        st.code(code, language='python')
    with col2:
        st.markdown("""##""")
        st.image("images/labeled.png")
        st.subheader("15 review for each category")

    st.divider()

    st.image("images/argilla.png")

    code = ''' 
    import argilla as rg
    
    g.init(workspace="admin")
    remote_dataset = review_df.push_to_argilla(name="resturent-reviews", workspace="admin")
    '''
    st.code(code, language='python')

    # run button
    if st.button(":running:"):
        st.session_state.valid = True
    
    if st.session_state.valid:
        components.iframe("http://localhost:6901/datasets?workspaces=argilla", height=1000, width=800)

        code2 = ''' 
        reviews_labeld_df = rg.load(self.rg_dataset_name, query="status: Validated").to_pandas()
        reviews_labeld_df.head(5)
        '''
        st.code(code2, language='python')

        # run button
        if st.button("run"):
            data_labeled = pd.read_excel("data/labeled_data.xlsx")
            st.dataframe(data_labeled[['text', 'annotation']].head(5))

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.image("images/good.png")
                st.write("- Multiple users can collaborate on labeling.")
                st.write("- Easy to install (provided as a Docker image).")
            with col2:
                st.image("images/bad.png")
                st.write("No issues encountered in my experience.")

# 6th slide (which is step 4 in the pipeline)
elif st.session_state.steps == 5:
    st.header('Training Few-shot model on our labeled reviews')
    st.markdown("""##""")

    st.image("images/setfit.png")
    st.subheader("SetFit: Efficient Few-Shot Learning Without Prompts")

    st.divider()

    code = '''
    from setfit import SetFitModel, SetFitTrainer
    from sentence_transformers.losses import CosineSimilarityLoss

    model_ul = SetFitModel.from_pretrained(transformer)
    # Create trainer
    trainer_ul = SetFitTrainer(
        model=model_ul,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        batch_size=16,
        num_iterations=20, # The number of text pairs to generate
    )

        # Train and evaluate
        trainer_ul.train()
    '''
    st.code(code, language='python')

    #run button
    if st.button(":running:"):
        HtmlFile = open("data/fig2.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height=1000, width=800)

        col1, col2 = st.columns(2)
        with col1:
            st.image("images/good.png")
            st.write("The accuracy is outstanding.")
        with col2:
            st.image("images/bad.png")
            st.write("Multilabel classification results in lower accuracy.")
# 7th slide 
elif st.session_state.steps == 6:
    st.header("Lets test the model and write some resturent reviews")
    st.markdown("""##""")

    col1, col2, col3 = st.columns([1,4, 1])
    with col2:
        st.image("images/qr_code.png")

    all_text = pd.DataFrame(columns=['Review'])

    # refresh button
    if st.button('Refresh!'):
        for path in os.listdir("share/"):
            data = pd.read_excel(f'share/{path}')[['Review']]
            text = data['Review'].values[0].replace(' ', '+')
            r = requests.get(f"http://127.0.0.1:8070/predict/{text}")
            res = json.loads(r.content)[text.replace('+', ' ')]
            if res[1] > 0.8:
                data['Predicted Category'] = res[0]
            else:
                data['Predicted Category'] = 'Others'
            all_text = pd.concat([all_text, data], axis=0)
            
        st.dataframe(all_text)


    


        
