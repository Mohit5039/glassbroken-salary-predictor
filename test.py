import requests

url = "https://job-salary-data.p.rapidapi.com/company-job-salary"
querystring = {
    "company": "Amazon",
    "job_title": "software developer",
    "location_type": "India",
    "years_of_experience": "2"
}
headers = {
    "x-rapidapi-host": "job-salary-data.p.rapidapi.com",
    "x-rapidapi-key": "a40d6b2fedmsh853fbbddb6f0efap178964jsn921d97fa9095"
}

response = requests.get(url, headers=headers, params=querystring)

# Print status
print("Status Code:", response.status_code)

# Write the JSON response to a text file for inspection
with open("api_output.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

print("Response written to api_output.txt")
