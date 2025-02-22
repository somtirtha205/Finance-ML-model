import json
import os
import ssl
import urllib.request


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(ssl, "_create_unverified_context", None):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script

# data = {"invoiceNumber":8828293477.0,"InvoiceAmount":67.9,"PaperlessBill":0.0,"InvoiceDate_Week":31.0,"InvoiceDate_IsWeekend":0.0,"DueDate_Week":35.0,"DueDate_IsWeekend":1.0,"InvoiceDate_count":21.0,"InvoiceAmount_sum":1272.4,"InvoiceAmount_mean":60.5904761905,"InvoiceAmount_max":88.48,"Disputed_mean":0.0952380952,"PaperlessBill_sum":12.0,"PaperlessBill_mean":0.5714285714,"DaysToSettle_mean":15.2380952381,"DaysToSettle_max":26.0,"DaysLate_mean":0.0,"DaysLate_max":0.0,"countryCode_391":0.0,"countryCode_406":0.0,"countryCode_770":1.0,"countryCode_818":0.0,"countryCode_897":0.0,"InvoiceDate_DOW_Friday":0.0,"InvoiceDate_DOW_Monday":0.0,"InvoiceDate_DOW_Saturday":0.0,"InvoiceDate_DOW_Sunday":0.0,"InvoiceDate_DOW_Thursday":1.0,"InvoiceDate_DOW_Tuesday":0.0,"InvoiceDate_DOW_Wednesday":0.0,"InvoiceDate_Month_April":0.0,"InvoiceDate_Month_August":1.0,"InvoiceDate_Month_December":0.0,"InvoiceDate_Month_February":0.0,"InvoiceDate_Month_January":0.0,"InvoiceDate_Month_July":0.0,"InvoiceDate_Month_June":0.0,"InvoiceDate_Month_March":0.0,"InvoiceDate_Month_May":0.0,"InvoiceDate_Month_November":0.0,"InvoiceDate_Month_October":0.0,"InvoiceDate_Month_September":0.0,"DueDate_DOW_Friday":0.0,"DueDate_DOW_Monday":0.0,"DueDate_DOW_Saturday":1.0,"DueDate_DOW_Sunday":0.0,"DueDate_DOW_Thursday":0.0,"DueDate_DOW_Tuesday":0.0,"DueDate_DOW_Wednesday":0.0,"DueDate_Month_April":0.0,"DueDate_Month_August":0.0,"DueDate_Month_December":0.0,"DueDate_Month_February":0.0,"DueDate_Month_January":0.0,"DueDate_Month_July":0.0,"DueDate_Month_June":0.0,"DueDate_Month_March":0.0,"DueDate_Month_May":0.0,"DueDate_Month_November":0.0,"DueDate_Month_October":0.0,"DueDate_Month_September":1.0}

with open("data/testing/sample_data.json", "r") as file:
    data = json.load(file)  # Load as dictionary

body = str.encode(json.dumps(data))

url = "http://823ac686-4f7b-4cd6-b660-95fe6341ea7a.eastus2.azurecontainer.io/score"


headers = {"Content-Type": "application/json"}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", "ignore"))
