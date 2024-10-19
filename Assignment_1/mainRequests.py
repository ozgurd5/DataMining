import requests
from bs4 import BeautifulSoup
import pandas as pd

# CF Clearance URL ve CycleTLS setup
CF_CLEARANCE_URL = 'http://localhost:3000/cf-clearance-scraper'

# WAF oturumu oluşturma fonksiyonu
def create_waf_session(url):
    try:
        print(f"WAF oturumu oluşturuluyor: {url}")
        response = requests.post(
            CF_CLEARANCE_URL,
            headers={'content-type': 'application/json'},
            json={'url': url, 'mode': 'waf-session'},
            timeout=20
        )

        if response.status_code == 200:
            print("WAF oturumu oluşturuldu.")
            data = response.json()
            return data.text

        else:
            print(f"WAF oturumu bağlantı hatası: {response.status_code}")
            create_waf_session(url)

    except Exception as e:
        print(f"Hata oluştu, tekrar deneniyor: {e}")
        create_waf_session(url)

# İlanları tutmak için bir liste
ilan_listesi = []

# İlanların linklerini tutmak için bir liste. Eşya ve doğalgaz durumlarına buradan ulaşacağız
ilan_linkleri = []

# 22 sayfa boyunca gezinme işlemi
for page in range(1, 2):
    url = f'https://www.hepsiemlak.com/buca-kiralik?page={page}'

    response = requests.get(url)

    while response.status_code != 200:
        print(f"Bağlantı hatası: {response.status_code}")
        header = create_waf_session(url)
        cookie_str = ""
        for cookie in header["cookies"]:
            cookie_str += f"{cookie["name"]}={cookie["value"]};"
        cookie_str = cookie_str[:-1]
        header["headers"]["cookie"] = cookie_str
        header["headers"]['user-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        response = requests.get(url, headers=header["headers"])

    # Sayfa kaynağını BeautifulSoup ile işleme
    soup = BeautifulSoup(response.content, "html.parser")
    ilanlar = soup.find_all("li", class_="listing-item")
    print(f"{len(ilanlar)} ilan bulundu.")

    # İlanları işleme
    for ilan in ilanlar:
        # Fiyat çıkarılıyor
        fiyat = ilan.find('span', class_='list-view-price').get_text(strip = True)
        print("Fiyat: ", fiyat)

        # İlan verileri ilana ekleniyor
        ilan_listesi.append({
            'Fiyat': fiyat,
        })

# İlan verilerini bir Excel dosyasına kaydetme
df = pd.DataFrame(ilan_listesi)
df.to_excel('ilanlar.xlsx', index=False)
print("İlanlar başarıyla kaydedildi: ilanlar.xlsx")
