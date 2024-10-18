import requests
from bs4 import BeautifulSoup
import pandas as pd

# CF Clearance URL ve CycleTLS setup
CF_CLEARANCE_URL = 'http://localhost:3000/cf-clearance-scraper'

# İlanları tutmak için bir liste
ilan_listesi = []

# İlanların linklerini tutmak için bir liste. Eşya ve doğalgaz durumlarına buradan ulaşacağız
ilan_linkleri = []

# WAF oturumu oluşturma fonksiyonu
def create_waf_session(url):
    try:
        print(f"WAF oturumu oluşturuluyor: {url}")
        response = requests.post(
            CF_CLEARANCE_URL,
            headers={'Content-Type': 'application/json'},
            json={'url': url, 'mode': 'waf-session'},  # WAF oturumu
            timeout=20
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 200:
                print("WAF oturumu oluşturuldu.")
                return data
            else:
                print(f"WAF oturumu oluşturulamadı: {data.get('message')}")
        else:
            print(f"WAF oturumu bağlantı hatası: {response.status_code}")
    except Exception as e:
        print(f"Hata oluştu: {e}")
    return None

# WAF oturumu ile sayfa kaynağını alma
def fetch_page_with_waf_session(url, session_data):
    try:
        # Oturumdan çerezleri ve başlıkları al
        cookies = session_data.get('cookies', [])
        headers = session_data.get('headers', {})

        # Çerezleri uygun formatta düzenle
        cookie_str = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies])
        headers['cookie'] = cookie_str

        # Bazı Cloudflare korumalı siteler User-Agent ve diğer başlıkları talep eder
        if 'user-agent' not in headers:
            headers['user-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'

        print(f"{url} sayfası isteniyor...")

        # requests ile sayfaya istek gönderme
        response = requests.get(
            url,
            headers=headers,
            timeout=20  # 20 saniye timeout ayarı
        )

        if response.status_code == 200:
            print(f"Sayfa başarıyla alındı: {url}")
            return response.text  # Sayfa kaynağını döndür
        else:
            print(f"Sayfa alınamadı: {response.status_code}")
    except Exception as e:
        print(f"Hata oluştu: {e}")
    return None

# 22 sayfa boyunca gezinme işlemi
for page in range(1, 2):
    url = f'https://www.hepsiemlak.com/buca-kiralik?page={page}'

    # WAF oturumu oluştur
    session_data = create_waf_session(url)
    if session_data:
        # WAF oturumu ile CycleTLS kullanarak sayfa kaynağını al
        page_content = fetch_page_with_waf_session(url, session_data)
        if page_content:
            # Sayfa kaynağını BeautifulSoup ile işleme
            soup = BeautifulSoup(page_content, "html.parser")
            ilanlar = soup.find_all("li", class_="listing-item")
            print(f"{len(ilanlar)} ilan bulundu.")

            # İlanları işleme
            # Her bir ilan kartını işleme
            for ilan in ilanlar:
                try:
                    # Fiyat çıkarılıyor
                    fiyat = ilan.find('span', class_='list-view-price').get_text(strip = True)
                    print("Fiyat: ", fiyat)

                    # İlan verileri ilana ekleniyor
                    ilan_listesi.append({
                        'Fiyat': fiyat,
                    })

                except Exception as e:
                    # Hata durumunda mesaj yazdırılıyor, işleme bir sonraki sayfadan devam ediliyor
                    print(f'Error: {e}')
                    continue
    else:
        print(f"{url} sayfası için WAF oturumu oluşturulamadı.")

# İlan verilerini bir Excel dosyasına kaydetme
df = pd.DataFrame(ilan_listesi)
df.to_excel('ilanlar.xlsx', index=False)
print("İlanlar başarıyla kaydedildi: ilanlar.xlsx")
