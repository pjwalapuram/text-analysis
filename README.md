# Text-Analysis API
[![Run on Google Cloud](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run)

## Text Analysis API can:

* **Scrape data from HTML and infer topics**
* **Save inferred topics, sentiments and documents to database**
* **Retrieve all such data**

### **/extract-themes**
* **Minimally expects URL path and can also take other parameters for topic model**
* **Automatically runs topic modeling and sentiment analysis**
* **Saves all data in sqlite3 database**

### **/return-themes**
* **Returns extracted themes**
* **Specify database name if passed durng extraction or use `text_analysis.db`**

### **/return-texts**
* **Returns documents, their sentiment and assigned themes**
* **Specify database name if passed durng extraction or use `text_analysis.db`**

### **/return-documents**
* **Returns list of extracted texts before processing**
* **Specify database name if passed durng extraction or use `text_analysis.db`**