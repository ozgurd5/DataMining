from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

# Start the WebDriver
driver = webdriver.Chrome()

# List to hold the ads
ilanlar = []

# Navigate through 20 pages
for page in range(1, 21):
    url = f'https://www.hepsiemlak.com/buca-kiralik?page={page}'
    driver.get(url)
    time.sleep(3)  # Wait for the page to load

    # Find the ad cards
    ilan_kartlari = driver.find_elements(By.CLASS_NAME, 'listing-item')

    for ilan in ilan_kartlari:
        try:
            # Extract price, title, location, number of rooms, area, and other information
            fiyat = ilan.find_element(By.CLASS_NAME, 'list-view-price').text
            fiyat = fiyat.replace("TL", "").replace(",", "").strip()  # Remove TL and commas, and strip extra spaces

            baslik = ilan.find_element(By.TAG_NAME, 'h3').text
            konum = ilan.find_element(By.CLASS_NAME, 'list-view-location').text

            # Extract additional details safely
            detay = ilan.find_element(By.CLASS_NAME, 'short-property').text.split("\n")

            oda_sayisi = detay[1] if len(detay) > 1 else ''  # Check length before accessing
            alan = detay[2] if len(detay) > 2 else ''
            bina_yasi = detay[3] if len(detay) > 3 else ''
            kat_sayisi = detay[4] if len(detay) > 4 else ''

            # Extract the date of the ad
            ilan_tarihi = ilan.find_element(By.CLASS_NAME, 'list-view-date').text

            # Add the ad data to the list
            ilanlar.append({
                'Fiyat': fiyat,
                'Başlık': baslik,
                'Konum': konum,
                'Oda Sayısı': oda_sayisi,
                'Alan': alan,
                'Bina Yaşı': bina_yasi,
                'Kat': kat_sayisi,
                'İlan Tarihi': ilan_tarihi
            })
        except Exception as e:
            print(f'Error: {e}')
            continue

# Close the browser
driver.quit()

# Convert the data to a DataFrame and save to an Excel file
df = pd.DataFrame(ilanlar)
df.to_excel('ilanlar.xlsx', index=False)
print("Ads successfully saved: ilanlar.xlsx")
