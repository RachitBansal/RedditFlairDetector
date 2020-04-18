# RedditFlairDetector
This project aims at identifying the Flair (Reddit terminology for _category_) of a given Submission (_post_) the r/India Subreddit.

## Tasks Undertaken:
<details><summary><b>1. Data Acquistion</b></summary>
  <p>
    Collected Data using <a href = "https://github.com/pushshift/api">pushshift.io</a>. Obtained a total of .7M data points across majorly 24 flairs. <br>
    <a href = "https://github.com/RachitBansal/RedditFlairDetector/blob/master/1_DataCollection.ipynb">Data Collection </a> <br>
    <a href = "https://github.com/RachitBansal/RedditFlairDetector/blob/master/2_EDA%26PreProcessing.ipynb">EDA and Pre-Processing </a>
  </p>
</details>
    
<details><summary><b>2. Data Modelling</b></summary>
  <p>
    Trained and Tested a wide range of models ranging from LogisticRegression to DistilBERT. <br>
    <a href = "https://github.com/RachitBansal/RedditFlairDetector/blob/master/3_Modelling_ML.ipynb">Machine Learning Models </a> <br>
    <a href = "https://github.com/RachitBansal/RedditFlairDetector/blob/master/3_Modelling_DL.ipynb">Deep Learning Models </a>
  </p>
</details>

<details><summary><b>3. Deploying on the Web</b></summary>
  <p>
    Designed the UI using HTML and CSS <br>
    Integrated with Flask and deployed on Heroku <br>
    <a href = "https://rflairdetection.heroku.com">The Website is LIVE</a>
  </p>
</details>

