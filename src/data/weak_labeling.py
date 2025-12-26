"""Weak labeling for distant supervision."""
import random
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
import re

from src.data.normalize_tr import normalize_turkish_text
from src.config import NUM_POSITIVE_EXAMPLES, NUM_NEGATIVE_EXAMPLES

logger = logging.getLogger(__name__)

# ============================================================================
# POZİTİF ŞABLONLAR - Deyim içeren cümleler için
# ============================================================================
TEMPLATES = [
    # === TEMEL ŞABLONLAR (Söz/deyim olarak kullanım) ===
    "Bugün yine {EXPR} ve kimse şaşırmadı.",
    "O an {EXPR} deyince ortam gerildi.",
    "Her zaman {EXPR} derdi büyükannem.",
    "{EXPR} sözü çok doğru bir söz.",
    "Bu durumda {EXPR} demek gerekiyor.",
    "O zaman {EXPR} dedi ve herkes güldü.",
    "Böyle durumlarda {EXPR} derler.",
    "{EXPR} diye bir söz vardır.",
    "O, {EXPR} sözünü çok severdi.",
    "Bazen {EXPR} demek yeterli olur.",
    "Bu konuda {EXPR} sözü geçerli.",
    "{EXPR} dediğinde herkes anladı.",
    "O gün {EXPR} demişti bana.",
    "Bu sözü duyunca {EXPR} geldi aklıma.",
    "{EXPR} sözü bu durumu çok iyi açıklıyor.",
    "Her zaman {EXPR} derdi annem.",
    "O an {EXPR} dedi ve herkes sustu.",
    "Bu tür durumlarda {EXPR} demek gerekir.",
    "{EXPR} sözü çok anlamlı bir söz.",
    "O zaman {EXPR} dedi ve herkes onayladı.",
    "Böyle zamanlarda {EXPR} derler.",
    "{EXPR} diye bilinen bir söz var.",
    "O, {EXPR} sözünü sık sık kullanırdı.",
    "Bazen {EXPR} demek en doğrusu olur.",
    "Bu konuda {EXPR} sözü çok uygun.",
    "{EXPR} dediğinde kimse itiraz etmedi.",
    "O gün {EXPR} demişti ve çok haklıydı.",
    "Bu sözü duyunca {EXPR} hatırladım.",
    "{EXPR} sözü bu durumu mükemmel açıklıyor.",
    "Her zaman {EXPR} derdi babam.",
    "O an {EXPR} dedi ve herkes düşündü.",
    "Bu tür durumlarda {EXPR} demek mantıklı.",
    "{EXPR} sözü çok değerli bir söz.",
    "O zaman {EXPR} dedi ve herkes beğendi.",
    "Böyle zamanlarda {EXPR} derler genelde.",
    "{EXPR} diye meşhur bir söz var.",
    "O, {EXPR} sözünü her fırsatta söylerdi.",
    "Bazen {EXPR} demek yeterli oluyor.",
    "Bu konuda {EXPR} sözü çok yerinde.",
    "{EXPR} dediğinde herkes başını salladı.",
    "O gün {EXPR} demişti ve çok doğruydu.",
    "Bu sözü duyunca {EXPR} aklıma geldi.",
    "{EXPR} sözü bu durumu harika açıklıyor.",
    "Her zaman {EXPR} derdi dedem.",
    "O an {EXPR} dedi ve herkes gülümsedi.",
    "Bu tür durumlarda {EXPR} demek doğru olur.",
    "{EXPR} sözü çok bilge bir söz.",
    "Böyle zamanlarda {EXPR} derler bazen.",
    "{EXPR} diye ünlü bir söz var.",
    "O, {EXPR} sözünü çok sever ve kullanırdı.",
    "Bazen {EXPR} demek en iyisi olur.",
    "Bu konuda {EXPR} sözü çok uygun düşüyor.",
    "{EXPR} dediğinde kimse karşı çıkmadı.",
    "Her zaman {EXPR} derdi ninem.",
    "O an {EXPR} dedi ve herkes dikkat kesildi.",
    "{EXPR} sözü çok önemli bir söz.",
    "Böyle zamanlarda {EXPR} derler genellikle.",
    "{EXPR} diye bilinen güzel bir söz var.",
    "O, {EXPR} sözünü sıkça kullanırdı.",
    "Her zaman {EXPR} derdi büyükbabam.",
    "O an {EXPR} dedi ve herkes sessizleşti.",
    "{EXPR} sözü çok değerli ve anlamlı bir söz.",
    "Böyle zamanlarda {EXPR} derler çoğu zaman.",
    "{EXPR} diye meşhur ve güzel bir söz var.",
    
    # === DOĞAL KULLANIM (Fiil çekimleriyle - GEÇMİŞ ZAMAN) ===
    "O {EXPR} ve sonra rahatladı.",
    "Dün {EXPR} ve çok mutlu oldu.",
    "Geçen hafta {EXPR} ve işler düzeldi.",
    "O zaman {EXPR} ve her şey yoluna girdi.",
    "Birkaç gün önce {EXPR} ve sonuç harikaydı.",
    "Geçen ay {EXPR} ve başarılı oldu.",
    "Dün akşam {EXPR} ve rahatladı.",
    "Geçen yıl {EXPR} ve çok iyi oldu.",
    "Bir süre önce {EXPR} ve memnun kaldı.",
    "Sabah {EXPR} ve günü iyi geçti.",
    "Öğleden sonra {EXPR} ve işleri halletti.",
    "Akşam {EXPR} ve dinlendi.",
    "Gece {EXPR} ve uyudu.",
    "Önceki gün {EXPR} ve başardı.",
    "Geçen pazar {EXPR} ve mutlu oldu.",
    "Dün öğlen {EXPR} ve yemek yedi.",
    "Geçen cumartesi {EXPR} ve eğlendi.",
    "Birkaç hafta önce {EXPR} ve sonuç aldı.",
    "Geçen sene {EXPR} ve çok sevindi.",
    "Dün gece {EXPR} ve rahat uyudu.",
    "Geçen cuma {EXPR} ve tatile çıktı.",
    "Birkaç ay önce {EXPR} ve işi bitirdi.",
    "Geçen pazartesi {EXPR} ve işe başladı.",
    "Dün sabah {EXPR} ve kahvaltı yaptı.",
    "Geçen salı {EXPR} ve toplantıya gitti.",
    "Birkaç yıl önce {EXPR} ve hayatı değişti.",
    "Geçen çarşamba {EXPR} ve alışverişe çıktı.",
    "Dün öğleden sonra {EXPR} ve arkadaşıyla buluştu.",
    "Geçen perşembe {EXPR} ve spor yaptı.",
    "Birkaç gece önce {EXPR} ve rüya gördü.",
    
    # === DOĞAL KULLANIM (Fiil çekimleriyle - ŞİMDİKİ ZAMAN) ===
    "Şu anda {EXPR} ve devam ediyor.",
    "Şimdi {EXPR} ve işler iyi gidiyor.",
    "Bu sırada {EXPR} ve her şey yolunda.",
    "Şu an {EXPR} ve sonuç bekleniyor.",
    "Hala {EXPR} ve çalışıyor.",
    "Şu sıralar {EXPR} ve başarılı oluyor.",
    "Bu aralar {EXPR} ve mutlu.",
    "Şu dakika {EXPR} ve bekliyor.",
    "Tam şu anda {EXPR} ve görüyor.",
    "Bu an itibarıyla {EXPR} ve ilerliyor.",
    "Şu saniye {EXPR} ve hissediyor.",
    "Bu süreçte {EXPR} ve öğreniyor.",
    "Şu aşamada {EXPR} ve gelişiyor.",
    "Bu dönemde {EXPR} ve büyüyor.",
    "Şu noktada {EXPR} ve anlıyor.",
    
    # === DOĞAL KULLANIM (Fiil çekimleriyle - GELECEK ZAMAN) ===
    "Yarın {EXPR} ve sonuç göreceğiz.",
    "Gelecek hafta {EXPR} ve işler düzelecek.",
    "Birkaç gün sonra {EXPR} ve başarılı olacak.",
    "Yakında {EXPR} ve her şey yoluna girecek.",
    "Gelecek ay {EXPR} ve planlar gerçekleşecek.",
    "Önümüzdeki hafta {EXPR} ve sonuç alacak.",
    "Birkaç saat sonra {EXPR} ve bitirecek.",
    "Yarın sabah {EXPR} ve başlayacak.",
    "Gelecek yıl {EXPR} ve hayatı değişecek.",
    "Önümüzdeki ay {EXPR} ve tamamlayacak.",
    "Birkaç dakika sonra {EXPR} ve görecek.",
    "Yarın akşam {EXPR} ve dinlenecek.",
    "Gelecek cuma {EXPR} ve tatile çıkacak.",
    "Önümüzdeki pazartesi {EXPR} ve işe başlayacak.",
    "Birkaç hafta sonra {EXPR} ve bitecek.",
    
    # === DOĞAL KULLANIM (MİŞ'Lİ GEÇMİŞ ZAMAN) ===
    "Meğer {EXPR} ve kimse bilmiyormuş.",
    "O {EXPR} miş de haberimiz yokmuş.",
    "Anlaşılan {EXPR} ve her şey düzelmiş.",
    "Görünüşe göre {EXPR} ve başarmış.",
    "Duyduğuma göre {EXPR} ve çok mutlu olmuş.",
    "Dediklerine göre {EXPR} ve işler yoluna girmiş.",
    "Söylendiğine göre {EXPR} ve sonuç almış.",
    "Rivayete göre {EXPR} ve herkes şaşırmış.",
    "Anlatılana göre {EXPR} ve çok sevinmiş.",
    "Nakledilene göre {EXPR} ve başarılı olmuş.",
    
    # === CÜMLE ORTASINDA KULLANIM ===
    "O kişi çok çalıştı, {EXPR} ve başarılı oldu.",
    "Bu konuda düşündü, {EXPR} ve karar verdi.",
    "Her zaman planlıydı, {EXPR} ve sonuç aldı.",
    "O gün hazırlandı, {EXPR} ve başardı.",
    "Bu iş için çalıştı, {EXPR} ve tamamladı.",
    "Sorunları çözmek için uğraştı, {EXPR} ve halletti.",
    "Hedeflere ulaşmak için çabaladı, {EXPR} ve başardı.",
    "Başarılı olmak için çalıştı, {EXPR} ve sonuç aldı.",
    "İşe gitti, {EXPR} ve eve döndü.",
    "Okula gitti, {EXPR} ve öğrendi.",
    "Marketten geldi, {EXPR} ve yemek yaptı.",
    "Erken kalktı, {EXPR} ve kahvaltı etti.",
    "Geç yattı, {EXPR} ve yorgun uyandı.",
    "Kitap okudu, {EXPR} ve bilgi edindi.",
    "Film izledi, {EXPR} ve eğlendi.",
    "Müzik dinledi, {EXPR} ve rahatladı.",
    "Spor yaptı, {EXPR} ve formda kaldı.",
    "Yemek yedi, {EXPR} ve doydu.",
    "Çay içti, {EXPR} ve sohbet etti.",
    "Telefon açtı, {EXPR} ve konuştu.",
    "Kapıyı çaldı, {EXPR} ve içeri girdi.",
    "Pencereyi açtı, {EXPR} ve hava aldı.",
    "Arabayı çalıştırdı, {EXPR} ve yola çıktı.",
    "Bilgisayarı açtı, {EXPR} ve çalışmaya başladı.",
    "Televizyonu kapattı, {EXPR} ve uyudu.",
    
    # === BAĞLAÇLARLA KULLANIM ===
    "{EXPR} ama yine de devam etti.",
    "{EXPR} çünkü öyle olması gerekiyordu.",
    "{EXPR} ve sonra rahatladı.",
    "{EXPR} fakat yine de umutlu kaldı.",
    "{EXPR} ancak pes etmedi.",
    "{EXPR} lakin yine de çalıştı.",
    "{EXPR} oysa ki başarılı oldu.",
    "{EXPR} halbuki sonuç iyiydi.",
    "{EXPR} ne var ki yine de devam etti.",
    "{EXPR} yalnız sonuç bekleniyor.",
    "{EXPR} hem de çok güzel oldu.",
    "{EXPR} üstelik beklenenden iyi.",
    "{EXPR} dahası herkes memnun kaldı.",
    "{EXPR} ayrıca başkaları da faydalandı.",
    "{EXPR} bunun yanı sıra ders de aldı.",
    "{EXPR} buna ek olarak tecrübe kazandı.",
    "{EXPR} dolayısıyla mutlu oldu.",
    "{EXPR} bu nedenle sevindi.",
    "{EXPR} bu yüzden rahatladı.",
    "{EXPR} sonuç olarak başardı.",
    "{EXPR} netice itibarıyla memnun kaldı.",
    "{EXPR} öyle ki herkes şaşırdı.",
    "{EXPR} o kadar ki kimse inanamadı.",
    "{EXPR} nitekim sonunda başardı.",
    "{EXPR} zira gerekiyordu.",
    
    # === SORU CÜMLELERİNDE KULLANIM ===
    "Neden {EXPR} diye sordu.",
    "Nasıl {EXPR} diye merak etti.",
    "Ne zaman {EXPR} diye düşündü.",
    "Kim {EXPR} diye araştırdı.",
    "Nerede {EXPR} diye aradı.",
    "Niçin {EXPR} diye sorguladı.",
    "Hangi durumda {EXPR} diye sordu.",
    "Ne şekilde {EXPR} diye öğrenmek istedi.",
    "Acaba {EXPR} mı diye düşündü.",
    "Gerçekten {EXPR} mi diye sordu.",
    "Hakikaten {EXPR} mı diye merak etti.",
    "Cidden {EXPR} mi diye sorguladı.",
    "Sahiden {EXPR} mı diye araştırdı.",
    
    # === OLUMSUZ KULLANIM ===
    "O {EXPR} değildi ama yine de başardı.",
    "Bu durumda {EXPR} olmadan devam etti.",
    "Her zaman {EXPR} olmazdı ama yine de çalışırdı.",
    "{EXPR} olmadı fakat yine de umutlu kaldı.",
    "{EXPR} yapamadı ama pes etmedi.",
    "{EXPR} edemedi lakin denemeye devam etti.",
    "{EXPR} başaramadı ancak öğrendi.",
    "{EXPR} göremedi fakat hissetti.",
    "{EXPR} anlayamadı yalnız çabaladı.",
    "{EXPR} bilmiyordu ama öğrendi.",
    "{EXPR} tanımıyordu fakat tanıdı.",
    "{EXPR} istemiyordu ama yaptı.",
    "{EXPR} beklemiyordu ancak oldu.",
    "{EXPR} düşünmüyordu lakin düşündü.",
    "{EXPR} planlamıyordu fakat planladı.",
    
    # === EMİR KİPİNDE KULLANIM ===
    "Lütfen {EXPR} ve sonuç görelim.",
    "Şimdi {EXPR} ve işleri halledelim.",
    "Hemen {EXPR} ve başlayalım.",
    "Birlikte {EXPR} ve tamamlayalım.",
    "Haydi {EXPR} ve görelim.",
    "Artık {EXPR} ve bitirelim.",
    "Gel {EXPR} ve yapalım.",
    "Bak {EXPR} ve anla.",
    "Dur {EXPR} ve dinle.",
    "Bekle {EXPR} ve gör.",
    "Sabret {EXPR} ve kazan.",
    "Dayan {EXPR} ve başar.",
    "Çalış {EXPR} ve eriş.",
    "Uğraş {EXPR} ve başar.",
    "Dene {EXPR} ve gör.",
    
    # === İSTEK KİPİNDE KULLANIM ===
    "Keşke {EXPR} ve başarılı olsak.",
    "İnşallah {EXPR} ve işler düzelsin.",
    "Umarım {EXPR} ve sonuç iyi olsun.",
    "Dilerim {EXPR} ve her şey yoluna girsin.",
    "Allah korusun {EXPR} olmasın.",
    "Maşallah {EXPR} olmuş.",
    "Elhamdülillah {EXPR} gerçekleşmiş.",
    "İnşallah {EXPR} olacak.",
    "Hayırlısı {EXPR} olsun.",
    "Nasip olursa {EXPR} olacak.",
    
    # === ŞART KİPİNDE KULLANIM ===
    "Eğer {EXPR} olursa herkes mutlu olur.",
    "Şayet {EXPR} gerçekleşirse çok iyi olur.",
    "{EXPR} olsaydı her şey farklı olurdu.",
    "{EXPR} yaparsan başarırsın.",
    "{EXPR} edersen kazanırsın.",
    "{EXPR} olursa göreceksin.",
    "{EXPR} yapmış olsaydın şimdi rahat ederdin.",
    "{EXPR} etmiş olsaydın pişman olmazdın.",
    "Madem {EXPR} o zaman devam et.",
    "Mademki {EXPR} öyleyse bekle.",
    
    # === KARŞILAŞTIRMA KULLANIMLARI ===
    "Herkes {EXPR} ama o farklı davrandı.",
    "Bazıları {EXPR} bazıları ise farklı düşündü.",
    "Kimisi {EXPR} kimisi ise karşı çıktı.",
    "Bir kısmı {EXPR} diğerleri ise bekledi.",
    "Çoğu kişi {EXPR} az kişi ise reddetti.",
    "Birçoğu {EXPR} birkaçı ise tereddüt etti.",
    "Pek çoğu {EXPR} azı ise vazgeçti.",
    
    # === ZAMAN ZARFLARIYLA KULLANIM ===
    "Her zaman {EXPR} ve başarılı olurdu.",
    "Bazen {EXPR} bazen ise farklı davranırdı.",
    "Nadiren {EXPR} ve şaşırırdı.",
    "Sık sık {EXPR} ve memnun kalırdı.",
    "Ara sıra {EXPR} ve eğlenirdi.",
    "Daima {EXPR} ve mutlu olurdu.",
    "Sürekli {EXPR} ve ilerleme kaydederdi.",
    "Genellikle {EXPR} ve sonuç alırdı.",
    "Çoğunlukla {EXPR} ve kazanırdı.",
    "Zaman zaman {EXPR} ve düşünürdü.",
    "Arada bir {EXPR} ve dinlenirdi.",
    "Her defasında {EXPR} ve şaşırırdı.",
    "Her seferinde {EXPR} ve öğrenirdi.",
    "Her zaman ki gibi {EXPR} ve devam etti.",
    
    # === MESLEK VE ROL BAĞLAMINDA KULLANIM ===
    "Öğretmen {EXPR} dedi ve öğrenciler dinledi.",
    "Doktor {EXPR} dedi ve hasta rahatladı.",
    "Avukat {EXPR} dedi ve dava kazanıldı.",
    "Mühendis {EXPR} dedi ve proje tamamlandı.",
    "Sanatçı {EXPR} dedi ve eser ortaya çıktı.",
    "Yazar {EXPR} dedi ve kitap yazıldı.",
    "Müdür {EXPR} dedi ve karar alındı.",
    "Başkan {EXPR} dedi ve herkes onayladı.",
    "Lider {EXPR} dedi ve takım motive oldu.",
    "Usta {EXPR} dedi ve çırak öğrendi.",
    "Patron {EXPR} dedi ve işler hızlandı.",
    "Şef {EXPR} dedi ve yemek pişti.",
    "Kaptan {EXPR} dedi ve gemi yola çıktı.",
    "Pilot {EXPR} dedi ve uçak kalktı.",
    "Hemşire {EXPR} dedi ve hasta iyileşti.",
    
    # === AİLE BAĞLAMINDA KULLANIM ===
    "Annem {EXPR} derdi ve haklı çıkardı.",
    "Babam {EXPR} derdi ve öğrenirdim.",
    "Dedem {EXPR} derdi ve dinlerdim.",
    "Ninem {EXPR} derdi ve anlardım.",
    "Ağabeyim {EXPR} derdi ve takip ederdim.",
    "Ablam {EXPR} derdi ve yardım ederdi.",
    "Kardeşim {EXPR} dedi ve güldük.",
    "Eşim {EXPR} dedi ve kabul ettim.",
    "Çocuğum {EXPR} dedi ve gülümsedim.",
    "Torunum {EXPR} dedi ve sevindim.",
    "Amcam {EXPR} derdi ve şaşırırdım.",
    "Dayım {EXPR} derdi ve eğlenirdim.",
    "Halam {EXPR} derdi ve öğrenirdim.",
    "Teyzem {EXPR} derdi ve dinlerdim.",
    "Kuzenin {EXPR} dedi ve güldük.",
    
    # === ARKADAŞLIK BAĞLAMINDA KULLANIM ===
    "Arkadaşım {EXPR} dedi ve güldük.",
    "Dostum {EXPR} dedi ve anlaştık.",
    "Komşum {EXPR} dedi ve yardım ettim.",
    "Tanıdığım {EXPR} dedi ve şaşırdım.",
    "Ahbabım {EXPR} dedi ve sohbet ettik.",
    "Yoldaşım {EXPR} dedi ve devam ettik.",
    "Kanka {EXPR} dedi ve eğlendik.",
    "Meslektaşım {EXPR} dedi ve tartıştık.",
    "İş arkadaşım {EXPR} dedi ve işe koyulduk.",
    "Okul arkadaşım {EXPR} dedi ve hatırladık.",
    
    # === GÜNLÜK HAYAT BAĞLAMLARI ===
    "Sabah kalktığımda {EXPR} olduğunu gördüm.",
    "Akşam eve geldiğimde {EXPR} fark ettim.",
    "Yemek yerken {EXPR} konuştuk.",
    "Yürüyüş yaparken {EXPR} düşündüm.",
    "Araba kullanırken {EXPR} hatırladım.",
    "Telefonda konuşurken {EXPR} söyledim.",
    "Kitap okurken {EXPR} aklıma geldi.",
    "Müzik dinlerken {EXPR} hissettim.",
    "Film izlerken {EXPR} anladım.",
    "Ders çalışırken {EXPR} öğrendim.",
    "İş yaparken {EXPR} fark ettim.",
    "Alışveriş yaparken {EXPR} gördüm.",
    "Temizlik yaparken {EXPR} buldum.",
    "Yemek yaparken {EXPR} denedim.",
    "Spor yaparken {EXPR} hissettim.",
    "Uyumadan önce {EXPR} düşündüm.",
    "Uyanır uyanmaz {EXPR} anladım.",
    "Eve girer girmez {EXPR} gördüm.",
    "Dışarı çıkar çıkmaz {EXPR} hissettim.",
    "İşe başlar başlamaz {EXPR} fark ettim.",
    
    # === DUYGU BAĞLAMLARI ===
    "Çok mutlu oldu, {EXPR} ve güldü.",
    "Üzgündü, {EXPR} ve ağladı.",
    "Kızgındı, {EXPR} ve bağırdı.",
    "Şaşırdı, {EXPR} ve dondu kaldı.",
    "Sevindi, {EXPR} ve kutladı.",
    "Endişelendi, {EXPR} ve düşündü.",
    "Rahatladı, {EXPR} ve nefes aldı.",
    "Heyecanlandı, {EXPR} ve zıpladı.",
    "Korku duydu, {EXPR} ve kaçtı.",
    "Merak etti, {EXPR} ve araştırdı.",
    "Umut etti, {EXPR} ve bekledi.",
    "Hayal kırıklığına uğradı, {EXPR} ve üzüldü.",
    "Gururlandı, {EXPR} ve paylaştı.",
    "Utandı, {EXPR} ve saklandı.",
    "Pişman oldu, {EXPR} ve özür diledi.",
    "Minnettar oldu, {EXPR} ve teşekkür etti.",
    "Neşelendi, {EXPR} ve şarkı söyledi.",
    "Sakinleşti, {EXPR} ve dinlendi.",
    "Gerginleşti, {EXPR} ve stres yaptı.",
    "Özlem duydu, {EXPR} ve aradı.",
    
    # === İŞ VE KARİYER BAĞLAMLARI ===
    "Toplantıda {EXPR} konuşuldu.",
    "Projede {EXPR} uygulandı.",
    "Sunumda {EXPR} anlatıldı.",
    "Raporda {EXPR} belirtildi.",
    "Görüşmede {EXPR} tartışıldı.",
    "Mülakata {EXPR} soruldu.",
    "İşe alımda {EXPR} değerlendirildi.",
    "Terfide {EXPR} dikkate alındı.",
    "Performans değerlendirmesinde {EXPR} öne çıktı.",
    "Takım çalışmasında {EXPR} önemliydi.",
    "Liderlik ederken {EXPR} gerekiyordu.",
    "Karar verirken {EXPR} düşünüldü.",
    "Strateji belirlerken {EXPR} göz önünde bulunduruldu.",
    "Hedef koyarken {EXPR} planlandı.",
    "Başarı için {EXPR} şarttı.",
    
    # === EĞİTİM BAĞLAMLARI ===
    "Derste {EXPR} öğretildi.",
    "Sınavda {EXPR} soruldu.",
    "Ödevde {EXPR} istendi.",
    "Konferansta {EXPR} anlatıldı.",
    "Seminerde {EXPR} tartışıldı.",
    "Atölyede {EXPR} uygulandı.",
    "Laboratuvarda {EXPR} denendi.",
    "Kütüphanede {EXPR} araştırıldı.",
    "Ders kitabında {EXPR} yazıyordu.",
    "Notlarda {EXPR} geçiyordu.",
    "Tezde {EXPR} savunuldu.",
    "Araştırmada {EXPR} bulundu.",
    "Akademik çalışmada {EXPR} incelendi.",
    "Öğrenme sürecinde {EXPR} önemliydi.",
    "Eğitim boyunca {EXPR} vurgulandı.",
    
    # === MEKAN BAĞLAMLARI ===
    "Evde {EXPR} oldu.",
    "İşte {EXPR} yaşandı.",
    "Okulda {EXPR} öğrenildi.",
    "Sokakta {EXPR} görüldü.",
    "Parkta {EXPR} konuşuldu.",
    "Kafede {EXPR} tartışıldı.",
    "Restoranda {EXPR} söylendi.",
    "Hastanede {EXPR} duyuldu.",
    "Mağazada {EXPR} anlatıldı.",
    "Markette {EXPR} yaşandı.",
    "Sinemada {EXPR} hatırlandı.",
    "Tiyatroda {EXPR} canlandırıldı.",
    "Müzede {EXPR} sergilendi.",
    "Kütüphanede {EXPR} okundu.",
    "Stadyumda {EXPR} bağırıldı.",
    "Otelde {EXPR} konuşuldu.",
    "Havalimanında {EXPR} duyuldu.",
    "Tren istasyonunda {EXPR} söylendi.",
    "Otobüste {EXPR} anlatıldı.",
    "Metroda {EXPR} düşünüldü.",
    
    # === HAVA VE DOĞA BAĞLAMLARI ===
    "Yağmur yağarken {EXPR} düşündü.",
    "Güneş açınca {EXPR} sevindi.",
    "Kar yağdığında {EXPR} hatırladı.",
    "Fırtına çıkınca {EXPR} endişelendi.",
    "Rüzgar estiğinde {EXPR} üşüdü.",
    "Sıcakta {EXPR} bunaldı.",
    "Soğukta {EXPR} titredi.",
    "Bahar gelince {EXPR} neşelendi.",
    "Yaz başlayınca {EXPR} planladı.",
    "Sonbahar olunca {EXPR} hüzünlendi.",
    "Kış gelince {EXPR} hazırlandı.",
    "Gece olunca {EXPR} düşündü.",
    "Gündüz olunca {EXPR} çalıştı.",
    "Şafak söküncе {EXPR} uyandı.",
    "Akşam olunca {EXPR} dinlendi.",
    
    # === SPOR VE AKTİVİTE BAĞLAMLARI ===
    "Maçta {EXPR} yaşandı.",
    "Antrenman da {EXPR} öğrenildi.",
    "Yarışta {EXPR} gösterildi.",
    "Koşarken {EXPR} hissetti.",
    "Yüzerken {EXPR} rahatladı.",
    "Bisiklet sürerken {EXPR} düşündü.",
    "Dağ tırmanırken {EXPR} zorlandı.",
    "Kamp yaparken {EXPR} öğrendi.",
    "Yürüyüş yaparken {EXPR} gördü.",
    "Dans ederken {EXPR} eğlendi.",
    "Yoga yaparken {EXPR} sakinleşti.",
    "Meditasyon yaparken {EXPR} anladı.",
    "Egzersiz yaparken {EXPR} güçlendi.",
    "Spor yaparken {EXPR} formda kaldı.",
    "Oyun oynarken {EXPR} kazandı.",
    
    # === YEMEK VE İÇECEK BAĞLAMLARI ===
    "Yemek yerken {EXPR} konuştu.",
    "Kahve içerken {EXPR} düşündü.",
    "Çay içerken {EXPR} sohbet etti.",
    "Kahvaltı yaparken {EXPR} planladı.",
    "Öğle yemeğinde {EXPR} tartıştı.",
    "Akşam yemeğinde {EXPR} paylaştı.",
    "Pasta yerken {EXPR} kutladı.",
    "Meyve yerken {EXPR} sağlıklı hissetti.",
    "Su içerken {EXPR} ferahladı.",
    "Yemek pişirirken {EXPR} denedi.",
    
    # === İLETİŞİM BAĞLAMLARI ===
    "Telefonda {EXPR} konuşuldu.",
    "Mesajda {EXPR} yazıldı.",
    "E-postada {EXPR} belirtildi.",
    "Mektupta {EXPR} anlatıldı.",
    "Sosyal medyada {EXPR} paylaşıldı.",
    "Haberlerde {EXPR} duyuruldu.",
    "Gazetede {EXPR} yazıldı.",
    "Dergide {EXPR} yayınlandı.",
    "Radyoda {EXPR} söylendi.",
    "Televizyonda {EXPR} gösterildi.",
    "Podcastte {EXPR} tartışıldı.",
    "Videoda {EXPR} anlatıldı.",
    "Blogda {EXPR} paylaşıldı.",
    "Forumda {EXPR} konuşuldu.",
    "Sohbette {EXPR} geçti.",
    
    # === TEKNOLOJİ BAĞLAMLARI ===
    "Bilgisayarda {EXPR} yapıldı.",
    "Telefonda {EXPR} görüldü.",
    "İnternette {EXPR} araştırıldı.",
    "Uygulamada {EXPR} kullanıldı.",
    "Programda {EXPR} yazıldı.",
    "Sistemde {EXPR} kuruldu.",
    "Veritabanında {EXPR} kaydedildi.",
    "Bulutta {EXPR} saklandı.",
    "Ağda {EXPR} paylaşıldı.",
    "Sunucuda {EXPR} çalıştırıldı.",
    
    # === FELSEFİ VE DERİN ANLAMLI KULLANIM ===
    "Hayatta {EXPR} öğrendim.",
    "Yaşamda {EXPR} deneyimledim.",
    "Tecrübeyle {EXPR} anladım.",
    "Zamanla {EXPR} kavradım.",
    "Yıllar sonra {EXPR} fark ettim.",
    "Olgunlaştıkça {EXPR} gördüm.",
    "Bilgelikle {EXPR} kabul ettim.",
    "Sabırla {EXPR} başardım.",
    "İnançla {EXPR} devam ettim.",
    "Umutla {EXPR} bekledim.",
    "Sevgiyle {EXPR} yaptım.",
    "Şükürle {EXPR} karşıladım.",
    "Alçakgönüllülükle {EXPR} öğrendim.",
    "Cesaretleе {EXPR} atıldım.",
    "Kararlılıkla {EXPR} sürdürdüm.",
]

# ============================================================================
# NEGATİF ŞABLONLAR - Deyim içermeyen cümleler için
# ============================================================================
NEGATIVE_TEMPLATES = [
    # === GÜNLÜK HAYAT CÜMLELERİ ===
    "Bugün yine normal bir gün geçti ve kimse şaşırmadı.",
    "O an bir şey söyleyince ortam gerildi.",
    "Her zaman böyle derdi büyükannem.",
    "Bu söz çok doğru bir söz.",
    "Bu durumda bir şey demek gerekiyor.",
    "O zaman bir şey dedi ve herkes güldü.",
    "Böyle durumlarda böyle derler.",
    "Böyle bir söz vardır.",
    "O, bu sözü çok severdi.",
    "Bazen bir şey demek yeterli olur.",
    "Bu konuda bu söz geçerli.",
    "Bir şey dediğinde herkes anladı.",
    "O gün bir şey demişti bana.",
    "Bu sözü duyunca bir şey geldi aklıma.",
    "Bu söz bu durumu çok iyi açıklıyor.",
    "Her zaman böyle derdi annem.",
    "O an bir şey dedi ve herkes sustu.",
    "Bu tür durumlarda bir şey demek gerekir.",
    "Bu söz çok anlamlı bir söz.",
    "O zaman bir şey dedi ve herkes onayladı.",
    
    # === SABAH RUTİNİ ===
    "Sabah erken kalktım ve kahvaltı yaptım.",
    "Sabah saat yedide uyandım ve duş aldım.",
    "Güne kahveyle başladım ve gazete okudum.",
    "Erken kalkıp spor yaptım ve enerji topladım.",
    "Sabah kahvaltısında yumurta ve peynir yedim.",
    "Günaydın dedim ve işe hazırlandım.",
    "Sabah rutinime sadık kaldım ve günü planladım.",
    "Erkenden kalktım ve meditasyon yaptım.",
    "Sabah çayımı içtim ve haber izledim.",
    "Güne pozitif başladım ve gülümsedim.",
    
    # === İŞE GİTME ===
    "İşe giderken trafikte kaldım.",
    "Otobüsle işe gittim ve kitap okudum.",
    "Arabamla işe gittim ve müzik dinledim.",
    "Metroya bindim ve işe vardım.",
    "İşe yürüyerek gittim ve temiz hava aldım.",
    "Tramvayla işe gittim ve manzarayı izledim.",
    "Bisikletle işe gittim ve spor yapmış oldum.",
    "Taksiyle işe gittim ve e-postalarımı kontrol ettim.",
    "Servisle işe gittim ve arkadaşlarımla sohbet ettim.",
    "İşe giderken podcast dinledim ve bilgi aldım.",
    
    # === ÖĞLE TATİLİ ===
    "Öğle yemeğinde salata yedim.",
    "Öğlen molasında dışarı çıktım ve yürüdüm.",
    "Kantinde yemek yedim ve arkadaşlarımla konuştum.",
    "Öğle arası kitap okudum ve dinlendim.",
    "Restoranda öğle yemeği yedim ve döndüm.",
    "Evden getirdiğim yemeği yedim ve tasarruf ettim.",
    "Öğlen kahve içtim ve enerji topladım.",
    "Öğle tatilinde alışveriş yaptım ve ihtiyaçlarımı aldım.",
    "Öğlen arkadaşımla buluştum ve sohbet ettik.",
    "Öğle arası kısa bir yürüyüş yaptım ve ferahladım.",
    
    # === AKŞAM EVE DÖNÜŞ ===
    "Akşam evde kitap okudum.",
    "Eve döndüm ve yemek hazırladım.",
    "Akşam televizyon izledim ve dinlendim.",
    "Eve geldim ve ailemle vakit geçirdim.",
    "Akşam yemeği yedim ve sohbet ettim.",
    "Eve vardım ve duş aldım.",
    "Akşam müzik dinledim ve rahatladım.",
    "Eve geldim ve temizlik yaptım.",
    "Akşam dışarı çıktım ve yürüyüş yaptım.",
    "Eve döndüm ve erken yattım.",
    
    # === HAFTA SONU AKTİVİTELERİ ===
    "Hafta sonu temizlik yaptım.",
    "Cumartesi günü alışveriş yaptım.",
    "Pazar günü dinlenmeyi tercih ediyorum.",
    "Hafta sonu sinemaya gittim ve film izledim.",
    "Cumartesi arkadaşlarımla buluştum ve eğlendik.",
    "Pazar günü ailemi ziyaret ettim ve birlikte yemek yedik.",
    "Hafta sonu kitap okudum ve dinlendim.",
    "Cumartesi spor yaptım ve formda kaldım.",
    "Pazar günü kahvaltıyı dışarıda yaptım ve manzaranın tadını çıkardım.",
    "Hafta sonu ev işlerini hallettim ve rahatladım.",
    "Cumartesi parkta piknik yaptık ve güzel vakit geçirdik.",
    "Pazar günü bahçe işleriyle uğraştım ve çiçekleri suladım.",
    "Hafta sonu yeni bir hobi denedim ve resim yaptım.",
    "Cumartesi konsere gittim ve müziğin tadını çıkardım.",
    "Pazar günü kahve dükkanında oturdum ve gazete okudum.",
    
    # === ALIŞVERİŞ ===
    "Dün akşam markete gittim ve ekmek aldım.",
    "Alışveriş yaparken liste hazırladım.",
    "Mağazadan yeni kıyafetler aldım ve eve döndüm.",
    "Marketten meyve ve sebze aldım ve sağlıklı beslendim.",
    "Alışveriş merkezinde dolaştım ve vitrinlere baktım.",
    "Online alışveriş yaptım ve kargomun gelmesini bekledim.",
    "Pazardan taze ürünler aldım ve eve getirdim.",
    "Elektronik mağazasından telefon kılıfı aldım.",
    "Kitapçıdan yeni bir roman aldım ve okumaya başladım.",
    "Eczaneden ilaç aldım ve eve döndüm.",
    
    # === YEMEK VE MUTFAK ===
    "Yemek yaparken tuzu fazla koydum.",
    "Yemek pişirdim ve ailemle yedik.",
    "Mutfakta yeni bir tarif denedim ve başarılı oldu.",
    "Bulaşıkları yıkadım ve temizledim.",
    "Kahvaltı hazırladım ve masayı kurdum.",
    "Akşam yemeği için makarna yaptım ve lezzetli oldu.",
    "Tatlı yaptım ve misafirlere ikram ettim.",
    "Salata hazırladım ve diyet yaptım.",
    "Çorba pişirdim ve sıcak sıcak içtim.",
    "Kek yaptım ve çay eşliğinde yedim.",
    "Yemek yapmayı öğreniyorum ve her gün pratik yapıyorum.",
    "Mutfakta yeni malzemeler denedim ve farklı tatlar keşfettim.",
    "Barbekü yaptık ve dışarıda yedik.",
    "Kahve demleddim ve kokusunun tadını çıkardım.",
    "Çay demledim ve misafirlere ikram ettim.",
    
    # === SPOR VE SAĞLIK ===
    "Spor yapmak sağlık için çok önemlidir.",
    "Spor salonuna gittim ve egzersiz yaptım.",
    "Koşuya çıktım ve kondisyonumu artırdım.",
    "Yüzme havuzuna gittim ve yüzdüm.",
    "Yoga yaptım ve esnekliğimi geliştirdim.",
    "Bisiklete bindim ve parkta turladım.",
    "Yürüyüş yaptım ve temiz hava aldım.",
    "Pilates yaptım ve kaslarımı güçlendirdim.",
    "Tenis oynadım ve eğlendim.",
    "Basketbol oynadık ve ter attık.",
    "Futbol maçı yaptık ve gol attım.",
    "Dağa tırmandım ve zirveye ulaştım.",
    "Kamp yaptık ve doğada vakit geçirdik.",
    "Balık tuttum ve nehir kenarında dinlendim.",
    "Kayak yaptım ve kar sporlarının tadını çıkardım.",
    
    # === EĞİTİM VE ÖĞRENİM ===
    "Ders çalıştım ve sınava hazırlandım.",
    "Ödev yaptım ve teslim ettim.",
    "Kitap okudum ve özet çıkardım.",
    "Sunum hazırladım ve sınıfta sundum.",
    "Arkadaşımla ders çalıştık ve konuştuk.",
    "Öğretmenle görüştüm ve bilgi aldım.",
    "Kütüphanede araştırma yaptım ve notlar aldım.",
    "Sınavdan geçtim ve çok mutlu oldum.",
    "Proje hazırladım ve başarılı oldum.",
    "Ders notlarını gözden geçirdim ve çalıştım.",
    "Yeni bir dil öğrenmeye başladım ve kelime ezberledim.",
    "Online kursa kaydoldum ve derslerimi takip ettim.",
    "Sertifika programına katıldım ve mezun oldum.",
    "Konferansa katıldım ve yeni bilgiler edindim.",
    "Seminere gittim ve uzmanları dinledim.",
    
    # === İŞ HAYATI ===
    "Toplantıya katıldım ve notlar aldım.",
    "Raporu hazırladım ve sundum.",
    "Müşteriyle görüştüm ve anlaştık.",
    "Projeyi tamamladım ve teslim ettim.",
    "Sunum yaptım ve soruları cevapladım.",
    "E-postaları kontrol ettim ve cevapladım.",
    "Telefon görüşmesi yaptım ve randevu aldım.",
    "Dosyaları düzenledim ve arşivledim.",
    "Bütçeyi hazırladım ve onaylattım.",
    "Ekiple toplantı yaptım ve kararlar aldık.",
    "İş görüşmesine gittim ve iyi geçti.",
    "Terfi aldım ve kutladım.",
    "Yeni bir projeye başladım ve heyecanlandım.",
    "Meslektaşlarımla işbirliği yaptım ve verimli olduk.",
    "Hedeflerimi belirledim ve çalışmaya başladım.",
    
    # === SOSYAL İLİŞKİLER ===
    "Arkadaşımla telefonda konuştum.",
    "Arkadaşımla buluştum ve sohbet ettik.",
    "Aileyi ziyaret ettim ve güzel vakit geçirdik.",
    "Komşularla tanıştım ve selam verdim.",
    "Doğum günü partisine gittim ve eğlendim.",
    "Düğüne katıldım ve dans ettim.",
    "Mezuniyet törenine gittim ve kutladım.",
    "Arkadaş toplantısı yaptık ve kahkahalar attık.",
    "Eski arkadaşımla karşılaştım ve hasret giderdik.",
    "Yeni insanlarla tanıştım ve arkadaşlıklar kurdum.",
    
    # === EVCİL HAYVANLAR ===
    "Köpeği gezdirdim ve eve döndüm.",
    "Kedimi besledim ve sevdim.",
    "Balıklarımın suyunu değiştirdim.",
    "Papağanımla konuştum ve eğlendim.",
    "Köpeğimi veterinere götürdüm ve aşılarını yaptırdım.",
    "Kedimin tırnaklarını kestim.",
    "Köpeğimi yıkadım ve tüylerini taradım.",
    "Kuşumu kafesinden çıkardım ve uçmasına izin verdim.",
    "Hamsterımın kafesini temizledim.",
    "Evcil hayvanıma yeni oyuncaklar aldım.",
    
    # === EV İŞLERİ ===
    "Odayı topladım ve düzenledim.",
    "Çamaşırları yıkadım ve astım.",
    "Ütü yaptım ve kıyafetleri astım.",
    "Buzdolabını düzenledim ve temizledim.",
    "Camları sildim ve parlatttım.",
    "Halıları süpürdüm ve tozlarını aldım.",
    "Yatakları düzelttim ve odayı havalandırdım.",
    "Banyoyu temizledim ve dezenfekte ettim.",
    "Mutfağı temizledim ve düzenledim.",
    "Çöpleri çıkardım ve poşetleri değiştirdim.",
    
    # === BAHÇE VE DIŞ MEKAN ===
    "Bahçede çiçek ektim ve suladım.",
    "Çimenleri biçtim ve bahçeyi düzenledim.",
    "Ağaçları budadım ve dalları topladım.",
    "Sebze bahçesine domates ektim.",
    "Bahçe mobilyalarını temizledim ve yerleştirdim.",
    "Havuzu temizledim ve suyunu değiştirdim.",
    "Balkonumu düzenledim ve çiçek koydum.",
    "Terasımızda oturdum ve manzaranın tadını çıkardım.",
    "Bahçede mangal yaptık ve yedik.",
    "Çiçeklerimi suladım ve gübreledim.",
    
    # === ULAŞIM VE SEYAHAT ===
    "Okula giderken otobüs kullandım.",
    "Tatilde denize gittim ve yüzdüm.",
    "Seyahate çıktım ve yeni yerler keşfettim.",
    "Uçakla yolculuk yaptım ve varış noktasına ulaştım.",
    "Trenle seyahat ettim ve manzarayı izledim.",
    "Araba kiraladım ve şehri gezdim.",
    "Gemi turuna katıldım ve denizin tadını çıkardım.",
    "Yurt dışına gittim ve farklı kültürleri deneyimledim.",
    "Kamp yerine gittik ve çadır kurduk.",
    "Günübirlik geziye çıktık ve piknik yaptık.",
    
    # === DOĞA VE HAVA DURUMU ===
    "Hava çok sıcak olduğu için dışarı çıkmadım.",
    "Kışın kar yağdığında çok mutlu oluyorum.",
    "Yağmur yağıyordu ve şemsiyemi açtım.",
    "Güneş açmıştı ve dışarı çıktım.",
    "Rüzgar esiyordu ve şapkamı tuttum.",
    "Sis vardı ve dikkatli araba kullandım.",
    "Fırtına vardı ve evde kaldım.",
    "Hava güzeldi ve parkta oturdum.",
    "Gökkuşağı çıktı ve fotoğraf çektim.",
    "Gün batımını izledim ve manzaranın tadını çıkardım.",
    
    # === MEVSIMLER ===
    "Yazın denize gitmek istiyorum.",
    "Sonbaharda yapraklar döküldü ve çok güzel görünüyordu.",
    "Kışın evde sıcak çorba içmek güzel.",
    "Bahar geldi ve çiçekler açtı.",
    "Yaz tatilinde ailemle tatile gittik.",
    "Sonbahar renklerini çok seviyorum.",
    "Kış aylarında kayak yapmak eğlenceli.",
    "Bahar temizliği yaptım ve evi düzenledim.",
    
    # === SANAT VE KÜLTÜR ===
    "Sinemaya gittim ve film izledim.",
    "Tiyatroya gittim ve oyun izledim.",
    "Konsere gittim ve müziğin tadını çıkardım.",
    "Müzeyi gezdim ve sergileri inceledim.",
    "Galeriyi ziyaret ettim ve tabloları gördüm.",
    "Opera izledim ve çok etkilendim.",
    "Bale izledim ve dansçıları takdir ettim.",
    "Kitap fuarına gittim ve yeni kitaplar aldım.",
    "Fotoğraf sergisine gittim ve sanatçıları tanıdım.",
    "Sokak sanatçılarını izledim ve alkışladım.",
    
    # === MÜZİK VE EĞLENCE ===
    "Müzik dinlemeyi çok seviyorum.",
    "Bu şarkıyı çok seviyorum.",
    "Gitar çalmayı öğreniyorum ve pratik yapıyorum.",
    "Piyano dersi alıyorum ve ilerliyorum.",
    "Şarkı söyledim ve eğlendim.",
    "Müzik festivaline gittim ve harika vakit geçirdim.",
    "Yeni bir albüm dinledim ve beğendim.",
    "Radyo dinledim ve güncel şarkıları keşfettim.",
    "Karaoke yaptık ve eğlendik.",
    "Dans ettik ve gece geç saatlere kadar eğlendik.",
    
    # === TEKNOLOJİ ===
    "Bilgisayarda iş yaptım ve bitirdim.",
    "Telefonda annemi aradım ve konuştum.",
    "Yeni bir uygulama indirdim ve kullandım.",
    "Bilgisayarımı güncelledim ve yeniden başlattım.",
    "Tabletimde oyun oynadım ve eğlendim.",
    "Akıllı saatimi şarj ettim ve taktım.",
    "Kulaklıklarımla müzik dinledim.",
    "Fotoğraf makinemle resimler çektim.",
    "Video çektim ve düzenledim.",
    "Sosyal medyada paylaşım yaptım ve beğeni aldım.",
    
    # === SAĞLIK ===
    "Doktora gittim ve muayene oldum.",
    "İlaçlarımı içtim ve dinlendim.",
    "Sağlık kontrolü yaptırdım ve sonuçları bekledim.",
    "Dişçiye gittim ve diş temizliği yaptırdım.",
    "Göz muayenesine gittim ve gözlük aldım.",
    "Kan testi yaptırdım ve sonuçlarım iyi çıktı.",
    "Fizik tedaviye gittim ve egzersiz yaptım.",
    "Masaj yaptırdım ve rahatladım.",
    "Sağlıklı beslendim ve su içtim.",
    "Uyku düzenimi düzenledim ve erken yattım.",
    
    # === FİNANS VE EKONOMİ ===
    "Faturaları ödedim ve rahatladım.",
    "Bankaya gittim ve işlemlerimi yaptım.",
    "Bütçe planı yaptım ve tasarruf ettim.",
    "Yatırım yaptım ve kazanç elde ettim.",
    "Kredi kartı borcumu ödedim.",
    "Maaşımı aldım ve hesabıma yatırdım.",
    "Vergi beyannamemi verdim.",
    "Sigorta poliçemi yeniledim.",
    "Para biriktirdim ve geleceğe yatırım yaptım.",
    "Alışveriş listemi kontrol ettim ve bütçeme uydum.",
    
    # === DUYGUSAL DURUMLAR ===
    "Bugün çok mutluydum ve gülümsedim.",
    "Biraz üzgündüm ama toparlandım.",
    "Heyecanlandım ve sabırsızlandım.",
    "Rahatladım ve derin nefes aldım.",
    "Şaşırdım ve inanamadım.",
    "Gururlandım ve paylaştım.",
    "Minnettardım ve teşekkür ettim.",
    "Umutlandım ve bekledim.",
    "Merak ettim ve araştırdım.",
    "Sakinleştim ve düşündüm.",
    
    # === PLANLAMA VE ORGANİZASYON ===
    "Yarın sabah erken kalkmam gerekiyor çünkü işe gitmem lazım.",
    "Bu işi yapmak için zaman ayırmam gerekiyor.",
    "Hafta içi yoğun bir programım var.",
    "Takvimime baktım ve randevularımı kontrol ettim.",
    "Liste yaptım ve görevlerimi sıraladım.",
    "Hedeflerimi belirledim ve planladım.",
    "Önceliklendirme yaptım ve önemli işlerle başladım.",
    "Zaman yönetimi yaptım ve verimli oldum.",
    "Toplantı planladım ve katılımcıları davet ettim.",
    "Proje planı hazırladım ve ekiple paylaştım.",
    
    # === GÖRÜŞ VE DÜŞÜNCELER ===
    "Bu konuda farklı düşünüyorum.",
    "Bu konuyu daha önce hiç duymamıştım.",
    "Bu konuyu anlamak kolay değil.",
    "Bu konuda daha fazla bilgi edinmem lazım.",
    "Bu konuda daha fazla araştırma yapmam gerekiyor.",
    "Fikrimi değiştirdim ve yeni bir bakış açısı kazandım.",
    "Kararsız kaldım ve biraz düşündüm.",
    "Karar verdim ve harekete geçtim.",
    "Görüşlerimi paylaştım ve tartıştık.",
    "Fikir alışverişi yaptık ve uzlaştık.",
    
    # === HOBİLER ===
    "Yeni bir hobi edinmek istiyorum.",
    "Resim yapmayı öğreniyorum ve eğleniyorum.",
    "Fotoğrafçılık yapıyorum ve anları yakalıyorum.",
    "El işi yapıyorum ve yaratıcılığımı kullanıyorum.",
    "Bahçecilik yapıyorum ve bitkilerle ilgileniyorum.",
    "Koleksiyon yapıyorum ve yeni parçalar ekliyorum.",
    "Satranç oynuyorum ve strateji geliştiriyorum.",
    "Bulmaca çözüyorum ve zeka egzersizi yapıyorum.",
    "Origami yapıyorum ve kağıttan şekiller oluşturuyorum.",
    "Makrame öğreniyorum ve dekoratif düğümler yapıyorum.",
    
    # === GÜNLÜK AKTİVİTELER ===
    "Bugün iyi bir gün geçirdim ve mutlu oldum.",
    "Güne güzel başladım ve verimli çalıştım.",
    "İşlerimi hallettim ve rahatladım.",
    "Günlük rutinime uydum ve düzenli oldum.",
    "Gün boyunca aktif kaldım ve enerjik hissettim.",
    "Akşama kadar çalıştım ve yoruldum.",
    "Günü değerlendirdim ve planlarımı yaptım.",
    "Bugün yeni şeyler öğrendim ve kendimi geliştirdim.",
    "Gün içinde molalar verdim ve dinlendim.",
    "Günü verimli geçirdim ve hedeflerime ulaştım.",
    
    # === TAMİRAT VE BAKIM ===
    "Arabamı servise götürdüm ve bakımını yaptırdım.",
    "Evdeki arızayı tamir ettim.",
    "Kapıyı yağladım ve gıcırtıyı giderdim.",
    "Ampulü değiştirdim ve odayı aydınlattım.",
    "Musluğu tamir ettim ve su kaçağını durdurdum.",
    "Bilgisayarımı temizledim ve hızlandırdım.",
    "Ayakkabımı boyattım ve parlatttım.",
    "Gözlüğümü temizledim ve net görmemi sağladım.",
    "Saatimin pilini değiştirdim.",
    "Telefonumun ekran koruyucusunu değiştirdim.",
    
    # === RANDEVULAR VE GÖRÜŞMELER ===
    "Randevuma gittim ve görüşmemi yaptım.",
    "Toplantıya katıldım ve fikirlerimi paylaştım.",
    "İş görüşmesine gittim ve sorulara cevap verdim.",
    "Danışmanlık aldım ve tavsiyeler dinledim.",
    "Avukatla görüştüm ve hukuki konuları konuştuk.",
    "Muhasebecimle görüştüm ve finansal durumumu inceledik.",
    "Emlakçıyla görüştüm ve evlere baktık.",
    "Öğretmenimle görüştüm ve derslerimi konuştuk.",
    "Antrenörümle görüştüm ve antrenman programımı ayarladık.",
    "Psikologla görüştüm ve hislerimi paylaştım.",
    
    # === KUTLAMALAR VE ÖZEL GÜNLER ===
    "Doğum günümü kutladım ve pasta kestim.",
    "Yılbaşını ailemle geçirdim ve hediyeler açtık.",
    "Bayramda akrabaları ziyaret ettim.",
    "Mezuniyetimi kutladım ve mutlu oldum.",
    "Evlilik yıl dönümümüzü kutladık ve romantik bir akşam geçirdik.",
    "Anneler gününde anneme çiçek aldım.",
    "Babalar gününde babamla vakit geçirdim.",
    "Sevgililer gününde sevdiğime hediye aldım.",
    "Emekliliğimi kutladım ve yeni döneme başladım.",
    "Terfi almamı kutladık ve şampanya açtık.",
]


def generate_positive_examples(lexicon: Dict[str, Dict], 
                               num_examples: int,
                               templates: List[str]) -> List[Dict]:
    """Generate positive examples by embedding idioms/proverbs into templates.
    
    Args:
        lexicon: Lexicon mapping normalized expressions to metadata.
        num_examples: Number of examples to generate.
        templates: List of sentence templates.
        
    Returns:
        List of dictionaries with text, label, expression, definition.
    """
    examples = []
    expressions = list(lexicon.keys())
    
    if not expressions:
        logger.warning("No expressions in lexicon for positive examples")
        return examples
    
    for _ in range(num_examples):
        # Random template
        template = random.choice(templates)
        
        # Random expression
        expr = random.choice(expressions)
        expr_original = lexicon[expr].get('original', expr)
        
        # Fill template
        text = template.format(EXPR=expr_original)
        
        examples.append({
            'text': text,
            'label': 1,
            'expression': expr_original,
            'definition': lexicon[expr].get('definition', '')
        })
    
    return examples


def extract_example_sentences_from_definition(definition: str) -> List[str]:
    """Extract example sentences from definition field.
    
    CSV'deki definition alanında HTML formatında örnek cümleler var:
    <i> 'Bunu başarmak için elinden geleni yapacaksın, dedi.' -</i>İ. O. Anar.
    
    Args:
        definition: Definition string that may contain example sentences.
        
    Returns:
        List of extracted example sentences.
    """
    if not definition or pd.isna(definition):
        return []
    
    sentences = []
    definition_str = str(definition)
    
    # Pattern 1: <i> 'cümle' -</i>Yazar. (en yaygın format)
    pattern1 = r"<i>\s*['\"]([^'\"]+)['\"]\s*-</i>"
    matches1 = re.findall(pattern1, definition_str, re.IGNORECASE | re.DOTALL)
    sentences.extend(matches1)
    
    # Pattern 2: <i>cümle</i> (tırnak olmadan)
    pattern2 = r"<i>([^<]+)</i>"
    matches2 = re.findall(pattern2, definition_str, re.IGNORECASE | re.DOTALL)
    for match in matches2:
        cleaned = re.sub(r'\s*-\s*[A-ZİĞÜŞÇÖ][^.]*\.?\s*$', '', match.strip())
        cleaned = re.sub(r"^['\"]|['\"]$", '', cleaned)
        if "'" not in cleaned and '"' not in cleaned and cleaned and len(cleaned) > 10:
            sentences.append(cleaned)
    
    # Pattern 3: 'cümle' format (without HTML)
    pattern3 = r"['\"]([^'\"]{15,})['\"]"
    matches3 = re.findall(pattern3, definition_str)
    sentences.extend(matches3)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        sent = re.sub(r'^[:\-]\s*', '', sent)
        sent = re.sub(r'\s*[:\-]\s*$', '', sent)
        if len(sent) >= 10 and len(sent) <= 200:
            if any(char.isalpha() for char in sent):
                cleaned_sentences.append(sent)
    
    return cleaned_sentences


def generate_examples_from_csv_definitions(df: pd.DataFrame, 
                                         expr_col: str, 
                                         def_col: str) -> List[Dict]:
    """Generate training examples from CSV definition field example sentences."""
    examples = []
    total_extracted = 0
    
    for _, row in df.iterrows():
        expr = str(row[expr_col]) if pd.notna(row[expr_col]) else ""
        definition = str(row[def_col]) if pd.notna(row[def_col]) else ""
        
        if not expr or not definition:
            continue
        
        example_sentences = extract_example_sentences_from_definition(definition)
        
        for example_sent in example_sentences:
            expr_normalized = normalize_turkish_text(expr)
            sent_normalized = normalize_turkish_text(example_sent)
            
            expr_words = set(expr_normalized.split())
            sent_words = set(sent_normalized.split())
            
            common_words = expr_words.intersection(sent_words)
            word_match = len(common_words) >= min(2, len(expr_words) // 2)
            
            if (expr_normalized in sent_normalized or 
                expr in example_sent or 
                word_match):
                examples.append({
                    'text': example_sent,
                    'label': 1,
                    'expression': expr,
                    'definition': definition
                })
                total_extracted += 1
    
    logger.info(f"Extracted {total_extracted} example sentences from CSV definitions")
    return examples


def augment_with_turkish_inflections(expr: str) -> List[str]:
    """Generate Turkish inflected forms of an expression.
    
    Türkçe'de ekler çok önemli. Deyimler farklı çekimlerle kullanılabilir.
    """
    inflected = [expr]
    
    # Geçmiş zaman ekleri
    past_suffixes = ['dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü']
    # Şimdiki zaman ekleri
    present_suffixes = ['yor', 'ıyor', 'iyor', 'uyor', 'üyor']
    # Gelecek zaman ekleri
    future_suffixes = ['acak', 'ecek']
    # Miş'li geçmiş ekleri
    misli_past_suffixes = ['mış', 'miş', 'muş', 'müş']
    
    if expr.endswith('mak') or expr.endswith('mek'):
        base = expr[:-3]
        for suffix in past_suffixes[:4]:
            inflected.append(base + suffix)
        for suffix in present_suffixes[:3]:
            inflected.append(base + suffix)
        for suffix in future_suffixes:
            inflected.append(base + suffix)
        for suffix in misli_past_suffixes[:2]:
            inflected.append(base + suffix)
    
    return list(set(inflected))


def generate_natural_positive_examples(lexicon: Dict[str, Dict],
                                      num_examples: int) -> List[Dict]:
    """Generate positive examples using idioms in natural sentence contexts."""
    examples = []
    expressions = list(lexicon.keys())
    
    if not expressions:
        logger.warning("No expressions in lexicon for natural examples")
        return examples
    
    # TEMPLATES zaten çok kapsamlı, onu kullan
    for _ in range(num_examples):
        expr = random.choice(expressions)
        expr_original = lexicon[expr].get('original', expr)
        
        # %40 orijinal, %60 çekimli versiyon (daha fazla çeşitlilik)
        if random.random() < 0.6:
            inflected_forms = augment_with_turkish_inflections(expr_original)
            expr_to_use = random.choice(inflected_forms)
        else:
            expr_to_use = expr_original
        
        context = random.choice(TEMPLATES)
        
        try:
            text = context.format(EXPR=expr_to_use)
        except KeyError:
            text = expr_to_use
        
        examples.append({
            'text': text,
            'label': 1,
            'expression': expr_original,
            'definition': lexicon[expr].get('definition', '')
        })
    
    return examples


def generate_negative_examples(num_examples: int,
                               templates: List[str]) -> List[Dict]:
    """Generate negative examples without idioms/proverbs."""
    examples = []
    
    for _ in range(num_examples):
        template = random.choice(templates)
        examples.append({
            'text': template,
            'label': 0,
            'expression': None,
            'definition': None
        })
    
    return examples


def generate_weak_labels(lexicon: Dict[str, Dict],
                        num_positive: int = NUM_POSITIVE_EXAMPLES,
                        num_negative: int = NUM_NEGATIVE_EXAMPLES,
                        use_natural_examples: bool = True) -> pd.DataFrame:
    """Generate weak labels for training using distant supervision."""
    logger.info(f"Generating {num_positive} positive and {num_negative} negative examples")
    
    if use_natural_examples:
        template_count = num_positive // 2
        natural_count = num_positive - template_count
        
        logger.info(f"  - {template_count} template-based examples")
        logger.info(f"  - {natural_count} natural context examples")
        
        positive_template = generate_positive_examples(lexicon, template_count, TEMPLATES)
        positive_natural = generate_natural_positive_examples(lexicon, natural_count)
        positive = positive_template + positive_natural
    else:
        positive = generate_positive_examples(lexicon, num_positive, TEMPLATES)
    
    negative = generate_negative_examples(num_negative, NEGATIVE_TEMPLATES)
    
    all_examples = positive + negative
    random.shuffle(all_examples)
    
    df = pd.DataFrame(all_examples)
    logger.info(f"Generated {len(df)} examples (positive: {sum(df['label']==1)}, negative: {sum(df['label']==0)})")
    
    return df
