import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import pickle  # used for saving & loading your trained ML model.
import time
# Loading our ML Models.
# model which will predict Heart Disease
heart_disease_predict_model = pickle.load(open("heart_disease_prediction_model.sav", 'rb'))
# model which will predict Diabetes
diabetes_predict_model = pickle.load(open("Diabetes_prediction_model.sav", 'rb'))


# For computing prediction of Heart Disease with the help of it's corresponding loaded trained model(i.e. heart_disease_predict_model)
def heart_disease_prediction(input_data):  # input_data are those datas which the user will give us as input for a Single Patient.

    # Converting the input_data from the user to a numpy array using  'asarray()' function
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshaping the numpy array ---------
    # We need reshaping which ensures that the model calculate the result for only a single patient
    # else the model will calculate the result for all the datas in the dataset
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # predicting value on the reshaped values
    # if 'target' == 0 patient doesn't have Heart diseases , else 'target' == 1 patient has Heart Diseases
    prediction = heart_disease_predict_model.predict(input_data_reshaped)  # prediction will store either 0 or 1, in the form of LIST
    print(prediction)

    if prediction[0] == 1:  # prediction[0], since 'prediction' is LIST variable.
        return "Patient has Heart Disease"
    else:
        return "Patient doesn't have any Heart Disease"


# For computing prediction of Diabetes with the help of it's corresponding loaded trained model(i.e. diabetes_predict_model)
def diabetes_prediction(input_data):
    input_as_numpy_array = np.asarray(input_data)
    reshaped_data = input_as_numpy_array.reshape(1, -1)

    # predicting value on the reshaped values
    # if 'target' == 0 patient doesn't have Diabetes , else 'target' == 1 patient has Diabetes
    predicted_value = diabetes_predict_model.predict(reshaped_data)
    return predicted_value



# main()
def main():
    with st.sidebar:
        
        selected = option_menu(menu_title="Dash Board",
                               options=["Home", "Heart Disease Prediction", "Diabetes prediction","Contact Developer"],
                               icons=["house", "heart-pulse", "activity","person-circle"], default_index=0,
                               menu_icon="list")

    # In above code while assigning icons, remember the order in which you have added the tabs(Home,HeartDisease,Diabetes)
    # Accordingly assign the icons corresponding to the  tabs in the proper order.
    # default_index means when the Webapp is loaded, in the Sidebar the 0th index/position tab will be ACTIVE.
    # index/positions: Home-> 0th, HeartDisease-> 1st, Diabetes-> 2nd


    #  If users selects 'Heart Disease prediction' tab in the sidebar
    if selected == "Heart Disease Prediction":
        st.title(body=":anatomical_heart: Heart Disease Prediction")
        col1, col2, col3 = st.columns(3)

        # -------- Getting the input data from the user ---------
        with col1:
            age = st.text_input(label="Age", placeholder="Age of the individual (years).")
            gender = st.text_input(label="Gender", placeholder="Male:1, Female:0")
            cholstrl = st.text_input(label="Cholesterol", placeholder="in mg/dL")
            bp = st.slider(label="Blood pressure (in mmHg)", min_value=90, max_value=179)
            hrt_rate = st.text_input(label="Heart rate", placeholder="Heart rate in beats per minute")

        with col2:
            smoking = st.text_input(label="Smoking status", placeholder="Never:0, Current:2, Former:2")
            alcohol_intake = st.text_input(label="Alcohol intake frequency", placeholder="None:0, Moderate:1, Heavy:2")
            exe_hrs = st.text_input(label="Exercise Hours", placeholder="Hours of exercise per week")
            family_Hist = st.text_input(label="Family history of heart disease", placeholder="Yes:1, No:0")
            diabetes = st.text_input(label="Diabetes status", placeholder="Yes:1, No:0")

        with col3:
            obesity = st.number_input(label="Obesity status", placeholder="Yes:1, No:0", min_value=0, max_value=1)
            stress_lvl = st.slider(label="Stress level",min_value=1, max_value=10)
            blood_sugar_lvl = st.text_input(label=" Fasting blood sugar leve", placeholder="(in mg/dL)")
            exercise_induced_angina = st.number_input(label="Exercise Induced Angina", min_value=0, max_value=1)
            chest_pain_typ = st.number_input(label="Chest Pain Type", placeholder=" (Typical Angina: 0/Atypical Angina: 1/Non-anginal Pain:2/Asymptomatic:3)", min_value=0,max_value=3)



        # ------- Code for Prediction --------
        diagnosis = " "

        # variable 'diagnosis' is currently Storing 'null value', which will later store
        # the predicted value from the 'def heart_disease_prediction(input_data)' function. '''

        # Creating a Button for Prediction: It will give you the Final Prediction result on the click of the Button you just created down below .
        if st.button('Heart Diagnosis Result'):  # creates a button with 'Heart Diagnosis Result' written on button.

            # taking user input and sending it for prediction to 'def heart_disease_prediction(input_data)' function.
            diagnosis = heart_disease_prediction([int(age), int(gender),int(cholstrl), bp,int(hrt_rate),int(smoking),int(alcohol_intake),int(exe_hrs),int(family_Hist),int(diabetes),
                                                  obesity,stress_lvl,int(blood_sugar_lvl),exercise_induced_angina,chest_pain_typ])
            # send the user input in the same order in which the columns are there in the dataset.

            # if 'target' == 0 patient doesn't have Heart diseases , else 'target' == 1 patient has Heart Diseases
            col1, col2  =st.columns(2)
            my_bar = col1.progress(0,)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text="Predicting....")
            time.sleep(1)
            my_bar.empty()

            if "doesn't" in diagnosis:
                col1.success(diagnosis, icon=":material/thumb_up:")
            else:
                col1.error(diagnosis, icon=":material/warning:")

            with col2:
                st.metric(label=":green[Accuracy score]",value="81﹪")


    #  If users selects 'Diabetes prediction' tab in the sidebar
    if selected == "Diabetes prediction":

        with st.form(key='diabetes_tab'):
            st.title("🩸 Diabetes Prediction")

            col1, col2 = st.columns(2)

            with col1:
                glucose_lvl = st.text_input(label="Glucose", placeholder="Glucose level in the blood (mg/dL)")
                bp = st.text_input(label="Blood Pressure", placeholder="Blood Pressure (mmHg)")
                skin_thick = st.text_input(label="skin Thickness", placeholder="Skin thickness (mu)")

            with col2:
                insulin_lvl = st.text_input(label="Insulin", placeholder="Insulin level in blood")
                bmi = st.number_input(label="BMI", min_value=0.0, max_value=67.1, step=0.1)
                age = st.text_input(label="Age", placeholder="Enter patient's age")

            result = " "  # initially Null value,  0: Not diabetes,  1: Diabetes

            if st.form_submit_button("Predict"):
                #  send the user input in the same order in which the columns are there in the dataset.
                result = diabetes_prediction([int(glucose_lvl), int(bp), int(skin_thick), int(insulin_lvl), bmi, int(age)])

                # if 'target' == 0 patient doesn't have Diabetes , else 'target' == 1 patient has Diabetes

                col1,col2 = st.columns(2)
                progress_bar = col1.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1,text="predicting....")
                time.sleep(1)
                progress_bar.empty()

                if result[0] == 1:
                    col1.error(body="Diabetes Found", icon=":material/warning:")
                else:
                    col1.success(body="Diabetes not Found", icon=":material/thumb_up:")

                with col2:
                    st.metric(label=":green[Accuracy score]",value="75.3﹪",)

    #  If users selects 'Home' tab in the sidebar
    if selected == "Home":
        st.title(body="Multiple Disease Prediction System")


        st.header(body="Project Overview",divider="red")
        st.markdown(body="This project aims to develop a web-based application that predicts the likelihood "
                         "of two major health risks: :red[**heart disease**] and :red[**diabetes**]. The app will utilize machine "
                         "learning models trained on datasets of patient information, including age, cholesterol "
                         "levels, smoking habits, stress levels, alcohol consumption, and glucose levels. "
                         "Users will input their personal data, and the app will generate personalized risk "
                         "assessments for both heart disease and diabetes. The goal is to provide an accessible "
                         "and user-friendly tool for individuals to understand their health risks and take proactive "
                         "steps towards prevention. The app is built using Streamlit and Python.")

        st.header(body="Heart Disease Prediction", divider="red")
        st.markdown(body="The Heart Disease prediction model uses :red[**Logistic Regression**] to predict the likelihood of a "
                         "patient having heart disease. The model takes into account various factors such as:")
        hrt_dis_factors = ['Age','Gender','Cholesterol level','Blood pressure','Alcohol frequency','Stress level',
                           'Smoking status','And many more day-to-day factors']
        col1, col2 = st.columns(2)
        with col1:
            for i in hrt_dis_factors:
                st.markdown("• " + i)

        with col2:
            container = st.container(border=True)
            container.subheader(body="Accuracy Score 🎯")
            container.metric(label=":red-background[Training Dataset]", value="83﹪")
            container.metric(label=":green-background[Testing Dataset]", value="81﹪")

        st.header(body="Diabetes Prediction", divider="red")
        st.markdown(body="The Diabetes prediction model also uses :red[**Logistic Regression**] to predict the likelihood of a "
                         "patient having diabetes. The model takes into account various factors such as:")
        diabetes_factors = ['Age', 'BMI', 'Glucose level', 'Blood pressure', 'Insulin level',
                            'And other relevant factors']
        col1, col2 = st.columns(2)
        with col1:
            for j in diabetes_factors:
                st.markdown("• " + j)

        with col2:
            container = st.container(border=True)
            container.subheader(body="Accuracy Score 🎯")
            container.metric(label=":red-background[Training Dataset]", value="77.6﹪")
            container.metric(label=":green-background[Testing Dataset]", value="75.3﹪")

        st.header(body="Machine Learning model ",divider="red")

        st.subheader(body="Logistic Regression: A Brief Overview")
        st.markdown(body="Logistic Regression is a type of supervised learning algorithm used for predicting the "
                         "outcome of a categorical dependent variable, based on one or more predictor variables. "
                         "In the context of the Multiple Disease Prediction project, Logistic Regression is used "
                         "to predict the likelihood of a patient having heart disease or diabetes.")

        st.markdown(body="In the Multiple Disease Prediction project, Logistic Regression is used to predict the "
                         "likelihood of a patient having heart disease or diabetes based on various day-to-day factors. "
                         "The algorithm is trained on a dataset and outputs a probability score, which is used to classify "
                         "the patient as having the disease or not. The model's performance is evaluated using accuracy "
                         "scores on both training and testing datasets.")

        st.subheader(body="How it Works")
        st.markdown(body="Logistic Regression works by analyzing the relationship between the predictor "
                         "variables (e.g. age, gender, cholesterol levels, etc.) and the dependent variable "
                         "(e.g. presence or absence of heart disease or diabetes). The algorithm learns the "
                         "patterns and relationships in the data and outputs a probability score, which indicates "
                         "the likelihood of the patient having the disease.")

        st.subheader(body="Why Logistic Regression?")
        st.markdown(body="Logistic Regression is a popular choice for this project because:")
        why_LR_used = ['It is well-suited for binary classification problems (e.g. presence or absence of a disease)',
                        'It can handle multiple predictor variables',
                        'It outputs a probability score, which can be used for classification',
                        'It is relatively easy to interpret and understand']
        for k in why_LR_used:
            st.markdown("• " + k)


        st.header(body="User Interface",divider="red")
        st.markdown(body="The project uses Streamlit to create a user-friendly interface for individuals to input "
                         "their data and receive predictions. The interface is designed to be intuitive and easy to "
                         "use, allowing users to quickly and easily assess their risk of developing heart disease and diabetes.")

        st.header(body="Project Goals",divider="red")
        st.markdown(body="The goals of this project are:")
        prj_goal_factors = ['To provide a user-friendly interface for individuals to assess their risk of developing heart disease and diabetes',
                            'To utilize machine learning models to predict the likelihood of these diseases based on various day-to-day factors',
                            'To achieve high accuracy scores on both training and testing datasets']
        for q in prj_goal_factors:
            st.markdown("• " + q)

        st.header(body="Conclusion", divider="red")
        st.markdown(body="The Multiple Disease Prediction project is a valuable tool for individuals to assess "
                         "their risk of developing heart disease and diabetes. By utilizing machine learning models "
                         "and a user-friendly interface, this project aims to provide accurate predictions and empower "
                         "individuals to take control of their health.")

    #  If users selects 'Contact Developer' tab in the sidebar
    if selected == "Contact Developer":

        container = st.container(border=True)
        col1,col2 = container.columns(2)

        col1.subheader(body="Debashish Maharana")
        col1.image(image="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif")
        col1.write("Email:  [maharanadenu@gmail.com](mailto:maharanadenu@gmail.com)")
        col1.write("GitHub: [Debashish-Maharana](https://github.com/Debashish-Maharana)")
        col1.write("Linkedin: [debashishmaharana](https://www.linkedin.com/in/debashishmaharana/)")
        col1.write("Twitter: [@debashish46310](https://x.com/debashish46310)")


        col2.image("https://user-images.githubusercontent.com/74038190/229223263-cf2e4b07-2615-4f87-9c38-e37600f8381a.gif")









if __name__ == '__main__':
    main()











