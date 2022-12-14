import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pydeck as pdk
from streamlit_option_menu import option_menu
import matplotlib.style as style
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium

FILENAME = "Boston Crimes.csv"
FILENAME_2 = "BostonDistricts.csv"

DATA_CODE = """   
        if selected == "Data":

        with st.sidebar:
            container = st.container()
            all_choices = st.checkbox("Select all")

            if all_choices:
                sort_by_crime = container.multiselect("Sort by crime:", crimes, crimes)
                sort_by_day = container.multiselect("Sort by day:", days_list, days_list)

            else:
                sort_by_crime = container.multiselect("Sort by crime:", crimes)
                sort_by_day = container.multiselect("Sort by day:", days_list)

        df3 = df.query('Crime == @sort_by_crime and Day == @sort_by_day')
        st.dataframe(df3)
        """

MAP_CODE = """
        if selected == "Map"
        
        with st.sidebar:
            container = st.container()
            all_choices = st.checkbox("Select all")

            if all_choices:
                sort_by_crime = container.multiselect("Sort by crime:", crimes, crimes)
                sort_by_day = container.multiselect("Sort by day:", days_list, days_list)

            else:
                sort_by_crime = container.multiselect("Sort by crime:", crimes)
                sort_by_day = container.multiselect("Sort by day:", days_list)

        df2 = df.query('Crime == @sort_by_crime and Day == @sort_by_day')
        df_map_2 = crimes_map_df(df2)
        st.map(df_map_2)
        """

BAR_CODE = """      
        if chart == 'Bar':
            fig, ax = plt.subplots()
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_xlabel('Day')
            ax.set_ylabel('# of Crimes')
            ax.set_title('Crimes by Day of the Week')
            colormap = plt.get_cmap('cividis')
            bars = ax.bar(range(len(data)), data, color=colormap(np.linspace(0, 1, len(data))))
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height, '%d' % int(height), ha='center', va='bottom')
            st.pyplot(fig)
            
            """

LINE_CODE = """
        if chart == 'Line':
            fig, ax = plt.subplots()
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_xlabel('Day')
            ax.set_ylabel('# of Crimes')
            ax.set_title('Crimes by Day of the Week')
            ax.plot(range(len(data)), data, color="blue")
            ax.plot(range(len(data)), data, "o")
            st.pyplot(fig)
            """

PIE_CODE = """
        if chart == 'Pie':
            fig, ax = plt.subplots()
            ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            """

PREDICTION_CODE = """
def predictive_model(df):

    df_predictive = df[['Crime', 'District', 'Month', 'Day', 'Hour']]
    le = LabelEncoder()
    df_predictive['District'] = le.fit_transform(df['District'])
    df_predictive['Day'] = le.fit_transform(df['Day'])
    df_predictive['Month'] = le.fit_transform(df['Month'])

    X = df_predictive.drop('Crime', axis=1)
    y = df_predictive['Crime']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    return model
    
model = predictive_model(df)
df_no_na = df.dropna()
df2 = pd.DataFrame().assign(District=df_no_na['District'],
                            Month=df_no_na['Month'],
                            Day=df_no_na['Day'],
                            Hour=df_no_na['Hour'])
st.title('Crime Predictor')
st.markdown('As seen below, the crime predictor takes 4 inputs: District, Month, Day, and Hour. It uses a '
            'logistic regression model to predict the type of crime that is most likely to occur given the '
            'four selections.')

district_name = st.selectbox('District', sorted(df2['District'].unique()))
month_name = st.selectbox('Month', sorted(df2['Month'].unique()))
day_name = st.selectbox('Day', sorted(df2['Day'].unique()))
hour = st.selectbox('Hour', sorted(df['Hour'].unique()))

districts = df2['District'].unique()
days = df2['Day'].unique()
months = df2['Month'].unique()

districts_dict = dictionary_maker(districts)
days_dict = dictionary_maker(days)
months_dict = dictionary_maker(months)

district = return_value(districts_dict, district_name)
month = return_value(months_dict, month_name)
day = return_value(days_dict, day_name)

prediction = model.predict([[district, month, day, hour]])
"""

SHOOTING_CODE = """
m = folium.Map(location=[42.3395419898301, -71.0694087696754], zoom_start=11)
    for i in range(len(df)):
        if df.iloc[i]["Shooting"] == "True":
            folium.Marker(
                location=[df.iloc[i]['Lat'], df.iloc[i]['Long']],
                popup=df.iloc[i]['Crime']
            ).add_to(m)
        st_data = st_folium(m)"""


def csv_read(filename):
    df = pd.read_csv(filename, header=0)

    return df


def remove_columns(df, column_1, column_2):

    del df[column_1], df[column_2]

    return df


def dataframe_cleaning(df):
    df["SHOOTING"] = df["SHOOTING"].replace({0: "False", 1: "True"})
    df["Location"] = df["Location"].replace({"(0, 0)": None})
    df["REPORTING_AREA"] = df["REPORTING_AREA"].replace({" ": None})
    df["Long"] = df["Long"].replace({0.0000: None})
    df["Lat"] = df["Lat"].replace({0.0000: None})
    df[['Date', 'Time']] = df.OCCURRED_ON_DATE.str.split(" ", expand=True)
    df = df.rename(columns={'INCIDENT_NUMBER': 'Incident Number',
                            'OFFENSE_CODE': 'Code',
                            'OFFENSE_DESCRIPTION': 'Crime',
                            'DISTRICT': 'District',
                            'REPORTING_AREA': 'Reporting Area',
                            'SHOOTING': 'Shooting',
                            'OCCURRED_ON_DATE': 'Remove Me',
                            'YEAR': 'Year',
                            'MONTH': 'Month',
                            'DAY_OF_WEEK': 'Day',
                            'HOUR': 'Hour',
                            'STREET': 'Street'})
    del df['Remove Me']

    df = df.dropna(axis=0, subset=['Long'])

    month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June'}
    df['Month'] = df['Month'].map(month_map)

    return df


def convert_districts(main_df, district_df):

    district_df.set_index('District', inplace=True)
    test = district_df.to_dict()['District Name']
    main_df['District'] = main_df['District'].map(test)

    return main_df


def crimes_map_df(df):

    df_map = pd.DataFrame().assign(lat=df['Lat'], lon=df['Long'], crime=df['Crime'])
    df_map = df_map.set_index('crime')

    return df_map


def predictive_model(df):

    df_predictive = df[['Crime', 'District', 'Month', 'Day', 'Hour']]
    le = LabelEncoder()
    df_predictive['District'] = le.fit_transform(df['District'])
    df_predictive['Day'] = le.fit_transform(df['Day'])
    df_predictive['Month'] = le.fit_transform(df['Month'])

    X = df_predictive.drop('Crime', axis=1)
    y = df_predictive['Crime']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    return model


def dictionary_maker(values_list):
    dictionary_name = {}
    x = 0
    for i in range(len(values_list)):
        dictionary_name[values_list[i]] = x
        x += 1

    return dictionary_name


def return_value(dictionary, key):
    return dictionary[key]


def on_button_clicked(code):
    st.code(code)


def convert_shooting_df(df):
    shooting_df = df.loc[df['Shooting'] == "True"]

    return shooting_df


def military_to_standard(time):
    if time == 0:
        time = "12 AM"
    elif time == 12:
        time = "12 PM"
    elif time >= 13:
        time -= 12
        time = str(time)
        time += " PM"
    else:
        time = str(time)
        time += " AM"

    return time


def main():

    df = csv_read(FILENAME)
    df = remove_columns(df, "OFFENSE_CODE_GROUP", "UCR_PART")
    df = dataframe_cleaning(df)
    df_district = csv_read(FILENAME_2)
    df = convert_districts(df, df_district)
    crimes = df['Crime'].unique()
    days_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    st.set_page_config(layout="wide")

    with st.sidebar:

        selected = option_menu(
            menu_title=None,
            options=["Home", "Data", "Map", "Charts", "Crime Prediction"],
            icons=["house", "graph-up", "map", "bar-chart", 'kanban'],
            orientation="vertical",
            styles={
                "icon": {"color": "yellow", "font-size": "25px"},
                "nav-link-selected": {"background-color": "purple"},
            }
        )

    if selected == "Home":
        st.title("Final Project")
        st.markdown("Hello! My name is Steven Pepe. This is my CS230 Final Project. It's a data analysis on 2021 "
                    "crimes in Boston. In this streamlit application you will find: Tables, Maps, Charts, and a "
                    "Predictive Model. Please use the sidebar to navigate yourself throughout my application. ")
        st.image("boston police.jpg")

    if selected == "Map":

        with st.sidebar:
            map_selection = st.selectbox("Select map", ["Locations", 'Shootings'])

        if map_selection == "Locations":
            st.title("Boston Crimes Map")
            st.markdown("The Boston Crimes map can be seen below. "
                        "The map can be changed using the sidebar.")

            with st.sidebar:
                container = st.container()
                all_choices = st.checkbox("Select all")

                if all_choices:
                    sort_by_crime = container.multiselect("Sort by crime:", crimes, crimes)
                    sort_by_day = container.multiselect("Sort by day:", days_list, days_list)

                else:
                    sort_by_crime = container.multiselect("Sort by crime:", crimes)
                    sort_by_day = container.multiselect("Sort by day:", days_list)

            df2 = df.query('Crime == @sort_by_crime and Day == @sort_by_day')
            df_map_2 = crimes_map_df(df2)
            st.map(df_map_2)

            if st.checkbox("Show code"):
                on_button_clicked(MAP_CODE)

        if map_selection == "Shootings":
            st.title("Boston Shootings Map")
            st.markdown("The map below displays all crimes where a shooting was involved.")
            m = folium.Map(location=[42.3395419898301, -71.0694087696754], zoom_start=11)
            for i in range(len(df)):
                if df.iloc[i]["Shooting"] == "True":
                    folium.Marker(
                        location=[df.iloc[i]['Lat'], df.iloc[i]['Long']],
                        popup=df.iloc[i]['Crime']
                    ).add_to(m)
            st_data = st_folium(m, width=1000, height=700)
            if st.checkbox("Show code"):
                on_button_clicked(SHOOTING_CODE)

    if selected == "Data":
        st.title("Boston Crimes DataFrame")
        st.markdown("The Boston Crimes DataFrame can be seen below. "
                    "The chart can be changed using the sidebar.")
        with st.sidebar:
            container = st.container()
            all_choices = st.checkbox("Select all")

            if all_choices:
                sort_by_crime = container.multiselect("Sort by crime:", crimes, crimes)
                sort_by_day = container.multiselect("Sort by day:", days_list, days_list)

            else:
                sort_by_crime = container.multiselect("Sort by crime:", crimes)
                sort_by_day = container.multiselect("Sort by day:", days_list)

        df3 = df.query('Crime == @sort_by_crime and Day == @sort_by_day')
        st.dataframe(df3)

        if st.checkbox("Show code"):
            on_button_clicked(DATA_CODE)

    if selected == "Charts":
        st.title("Boston Crimes Chart")
        df3 = df.Day.value_counts()
        dict_3 = df3.to_dict()
        days = list(dict_3.items())
        new_dict = {}

        for i in range(len(days_list)):
            for x in range(len(days)):
                if days_list[i] == days[x][0]:
                    new_dict[days[x][0]] = days[x][1]

        labels = list(new_dict.keys())
        data = list(new_dict.values())

        with st.sidebar:
            chart_info = st.selectbox("Select what information you want to display:", ["Entire DataFrame", "Shootings"])
            chart = st.selectbox("Select Chart Style:", ['Bar', 'Line', 'Pie'])

        if chart_info == "Entire DataFrame":
            if chart == 'Bar':
                st.markdown("A bar chart displaying the amount of crimes that occurred on each day can be seen below."
                            " You can change the map type using the sidebar.")
                fig, ax = plt.subplots()
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(labels, rotation=90)
                ax.set_xlabel('Day')
                ax.set_ylabel('# of Crimes')
                ax.set_title('Crimes by Day of the Week')
                colormap = plt.get_cmap('cividis')
                bars = ax.bar(range(len(data)), data, color=colormap(np.linspace(0, 1, len(data))))
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height, '%d' % int(height), ha='center', va='bottom')
                st.pyplot(fig)

                if st.checkbox("Show code"):
                    on_button_clicked(BAR_CODE)

            if chart == 'Line':
                st.markdown("A line chart displaying the amount of crimes that occurred on each day can be seen below."
                            " You can change the map type using the sidebar.")
                fig, ax = plt.subplots()
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(labels, rotation=90)
                ax.set_xlabel('Day')
                ax.set_ylabel('# of Crimes')
                ax.set_title('Crimes by Day of the Week')
                ax.plot(range(len(data)), data, color="blue")
                ax.plot(range(len(data)), data, "o")
                st.pyplot(fig)

                if st.checkbox("Show code"):
                    on_button_clicked(LINE_CODE)

            if chart == 'Pie':
                st.markdown("A pie chart displaying the amount of crimes that occurred on each day can be seen below."
                            " You can change the map type using the sidebar.")
                fig, ax = plt.subplots()
                ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

                if st.checkbox("Show code"):
                    on_button_clicked(PIE_CODE)

        if chart_info == "Shootings":
            df = convert_shooting_df(df)
            df3 = df.Day.value_counts()
            dict_3 = df3.to_dict()
            days = list(dict_3.items())
            new_dict = {}

            for i in range(len(days_list)):
                for x in range(len(days)):
                    if days_list[i] == days[x][0]:
                        new_dict[days[x][0]] = days[x][1]

            labels = list(new_dict.keys())
            data = list(new_dict.values())
            if chart == 'Bar':
                st.markdown("A bar chart displaying the amount of shooting related crimes "
                            "that occurred on each day can be seen below."
                            " You can change the map type using the sidebar.")
                fig, ax = plt.subplots()
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(labels, rotation=90)
                ax.set_xlabel('Day')
                ax.set_ylabel('# of Crimes')
                ax.set_title('Crimes by Day of the Week')
                colormap = plt.get_cmap('cividis')
                bars = ax.bar(range(len(data)), data, color=colormap(np.linspace(0, 1, len(data))))
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height, '%d' % int(height), ha='center', va='bottom')
                st.pyplot(fig)

            if chart == 'Line':
                st.markdown("A line chart displaying the amount of shooting related crimes "
                            "that occurred on each day can be seen below."
                            " You can change the map type using the sidebar.")
                fig, ax = plt.subplots()
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(labels, rotation=90)
                ax.set_xlabel('Day')
                ax.set_ylabel('# of Crimes')
                ax.set_title('Crimes by Day of the Week')
                ax.plot(range(len(data)), data, color="blue")
                ax.plot(range(len(data)), data, "o")
                st.pyplot(fig)

                if st.checkbox("Show code"):
                    on_button_clicked(LINE_CODE)

            if chart == 'Pie':
                st.markdown("A pie chart displaying the amount of shooting related crimes "
                            "that occurred on each day can be seen below."
                            " You can change the map type using the sidebar.")
                fig, ax = plt.subplots()
                ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

                if st.checkbox("Show code"):
                    on_button_clicked(PIE_CODE)


    if selected == "Crime Prediction":
        model = predictive_model(df)
        df_no_na = df.dropna()
        df2 = pd.DataFrame().assign(District=df_no_na['District'],
                                    Month=df_no_na['Month'],
                                    Day=df_no_na['Day'],
                                    Hour=df_no_na['Hour'])
        st.title('Crime Predictor')
        st.markdown('As seen below, the crime predictor takes 4 inputs: District, Month, Day, and Hour. It uses a '
                    'logistic regression model to predict the type of crime that is most likely to occur given the '
                    'four selections.')

        district_name = st.selectbox('District', sorted(df2['District'].unique()))
        month_name = st.selectbox('Month', sorted(df2['Month'].unique()))
        day_name = st.selectbox('Day', sorted(df2['Day'].unique()))
        hour = st.selectbox('Hour', sorted(df['Hour'].unique()))

        districts = df2['District'].unique()
        days = df2['Day'].unique()
        months = df2['Month'].unique()

        districts_dict = dictionary_maker(districts)
        days_dict = dictionary_maker(days)
        months_dict = dictionary_maker(months)

        district = return_value(districts_dict, district_name)
        month = return_value(months_dict, month_name)
        day = return_value(days_dict, day_name)

        prediction = model.predict([[district, month, day, hour]])

        hour = military_to_standard(hour)

        st.write(f"The predicted crime in {district_name} on a {day_name} at {hour} is {str.lower(prediction[0])}.")

        if st.checkbox("Show code"):
            on_button_clicked(PREDICTION_CODE)


if __name__ == "__main__":
    main()
