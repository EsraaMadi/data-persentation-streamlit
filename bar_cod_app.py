import streamlit as st
import pandas as pd 
import random

t = st.text_input(label=" ", placeholder="Write a resturent review")
if st.button('OK'):
    input_ = pd.DataFrame(data={'Review' : t}, index=[0])
    input_.to_excel(f'share/File{random.randint(0,90000)}.xlsx')
    st.success('Thanks!', icon="✅")
    