# NBABlowoutPredictor
## Project to predict the likelihood of a given NBA matchup being a blowout

The goal of the project is to create an app that consumers can use to gauge whether they want to attend a given NBA game. This comes from the idea that blowouts are generally not fun to attend, and, if given the choice, people would much rather attend a close game than a very lopsided one. The project has a secondary use, which is the ability to be used by NBA front offices to assess the difficulty of future games. They could use informtion from this project to better allocate film and prepare better.

We used data from NBA_stats API stored in an S3 bucket to train a model, which is then stored in the same S3 bucket. The predict.py file calls this model to pull data and make predictions.


## 📊 Data Source 
The data comes from NBA.com stats via the NBA_api stats wrapper. The API is free and simple to use. The data includes games pulled as far back as specified, and player data from specific seasons as specified. 

## 🛠️ Tools and Technologies Used
- **Languages:** *Python*
- **Libraries:** *streamlit, pandas, , SciPy, joblib, boto3, numpy, sklearn*
- **IDE:** *VS Code*
- **Simple Storage Service**

## 🚀 How to Run the Project
1. Clone the repository or copy the code into a Jupyter Notebook or Python file (Only predict.py and app.py are necessary to run the project)
2. Install the required libraries;
   ```bash
   pip install streamlit, pandas, boto3, joblib, numpym etc.
3. Run the streamlit app
   ```bash
   streamlit run app.py
You can also run the uploaded streamlit app from this link https://nbablowoutpredictor-htnarsuqsjczenfumtx6uh.streamlit.app/ 
   
