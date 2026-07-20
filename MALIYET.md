# Eğitim Maliyet Rehberi

Bu rehber, nanosohbet reçetesiyle sıfırdan bir Türkçe dil modeli eğitmenin **gerçekçi
maliyetini** açıklar. Rakamlar, bu repodaki gerçek eğitim koşularından ölçülmüştür.

> Kısa özet: Konuşabilen, akıcı Türkçe üreten küçük bir model **birkaç yüz dolara**,
> daha yetkin bir model ise **birkaç bin dolara** eğitilebilir. Aşağıda ayrıntılar var.

---

## Ölçekler

nanochat mimarisinde model boyutu `--depth` (d) ile belirlenir. Bu repoda eğittiğimiz
ve ölçtüğümüz iki ölçek:

| Ölçek | Yaklaşık parametre | Eğitim verisi | Ne yapabilir |
|:---|:---:|:---:|:---|
| **d20** | ~560 milyon | ~5,2 milyar token | Akıcı Türkçe üretir; temel sohbet |
| **d26** | ~2 milyar | ~11,6 milyar token | Daha tutarlı, daha bilgili; belirgin kalite sıçraması |

Daha büyük ölçekler (d32+) mümkündür ancak donanım ve süre gereksinimi hızla artar.

---

## Ölçülen performans (4×A100 80GB)

Bu repodaki ön-eğitim koşularından elde edilen gerçek değerler:

| Ölçek | Süre (4×A100) | GPU-saat | Verim (MFU) |
|:---|:---:|:---:|:---:|
| **d20** | ~8 saat | ~32 GPU-saat | ~%47 |
| **d26** | ~33 saat | ~132 GPU-saat | ~%47 |

*GPU-saat = süre × GPU sayısı. Bulut maliyetini bu sayı belirler.*

---

## Bulut maliyeti (yaklaşık)

A100 80GB kiralama fiyatı sağlayıcıya göre değişir; tipik aralık **saat başına
1,5–2,5 USD**'dir (spot/topluluk bulutlarında daha düşük olabilir).

| Ölçek | GPU-saat | Tahmini maliyet (A100) |
|:---|:---:|:---:|
| **d20** (ön-eğitim) | ~32 | **~50–80 USD** |
| **d26** (ön-eğitim) | ~132 | **~200–330 USD** |
| Tokenizer eğitimi | CPU (birkaç saat) | ~ihmal edilebilir |
| Talimat ayarı (SFT) | ~10–20 GPU-saat | ~15–50 USD |

> Örnek: **d20 ölçeğinde konuşabilen bir Türkçe model**, ön-eğitim + SFT dahil
> **~100 USD** civarında bir bütçeyle eğitilebilir.

Maliyeti düşüren etkenler: spot/kesintili GPU kullanımı, daha küçük derlem, daha kısa
eğitim. Artıran etkenler: daha büyük model, daha çok token, daha uzun eğitim.

---

## Kendi bütçenize göre yol

- **~100 USD:** d20 ölçeği — akıcı Türkçe üreten, sohbet edebilen bir model. Öğrenmek,
  denemek ve kavramı görmek için ideal.
- **~300–400 USD:** d26 ölçeği — belirgin kalite artışı; ciddi bir Türkçe temel model.
- **Kurumsal / araştırma:** Daha büyük ölçekler ve daha zengin veri hattı (hukuk, güncel
  olaylar, alan uzmanlığı) ile amiral modeller. Bu reçeteyle eğitilen
  [Erk](https://github.com/ecloudtechnology/erk), TurkishMMLU'da açık Türkçe modellerin
  birincisi oldu.

---

## Notlar

- Rakamlar yaklaşıktır ve donanım, sağlayıcı, veri hazırlığı ile değişir.
- Tek GPU ile de eğitim mümkündür; süre GPU sayısıyla ters orantılı uzar.
- Ön-eğitim verisi (Türkçe web derlemi) açık kaynaklardan indirilir; veri maliyeti
  yalnızca depolama ve indirme bant genişliğidir.
- Bu rehber ön-eğitim + talimat ayarını kapsar; değerlendirme ve çıkarım maliyetleri
  görece düşüktür.

---

*Bir [eCloud Yazılım Teknolojileri](https://www.e-cloud.web.tr) projesidir.*
