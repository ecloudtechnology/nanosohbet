<div align="center">

# 🐦 nanosohbet

### Kendi Türkçe dil modelini sıfırdan eğit

Tokenizer'dan sohbet arayüzüne kadar — tam, açık ve tekrarlanabilir bir reçete.
Aynı Türkçe veri ve eğitim yaklaşımını, amiral modelimiz
**[Erk](https://github.com/ecloudtechnology/erk)** için de kullandık; Erk (Qwen3-14B
tabanlı, Türkçe sürekli ön-eğitim + talimat ayarı) TurkishMMLU'da açık Türkçe modellerin
**birincisi** oldu.

<br>

[![Erk Model](https://img.shields.io/badge/🤗_Erk--14B-%25_69.7_TurkishMMLU-2ea043?style=for-the-badge)](https://huggingface.co/ecloudtech/Erk-14B)
[![Erk Repo](https://img.shields.io/badge/Model-Erk-c9a227?style=for-the-badge)](https://github.com/ecloudtechnology/erk)
[![Lisans](https://img.shields.io/badge/Lisans-MIT-111?style=for-the-badge)](LICENSE)

[Erk modelini indir (🤗)](https://huggingface.co/ecloudtech/Erk-14B) · **[Türkçe](#türkçe)** · **[English](#english)** · [eCloud Tech.](https://www.e-cloud.web.tr)

</div>

---

## Türkçe

### nanosohbet nedir?

**nanosohbet**, sıfırdan bir Türkçe dil modeli eğitmenin **uçtan uca reçetesidir.**
Amaç hazır bir model dağıtmak değil — **herkesin, makul bir bütçeyle, kendi Türkçe dil
modelini eğitebilmesini** sağlamaktır.

Türkçe için model eğitmek isteyen bir araştırmacı, öğrenci ya da şirket, bugün ya devasa
kapalı modellere bağımlı ya da dağınık, eksik kaynaklarla baş başa. nanosohbet bu boşluğu
doldurur: tokenizer eğitiminden ön-eğitime, talimat ayarından (SFT) sohbet arayüzüne
kadar **her adım tek bir kod tabanında, açık ve tekrarlanabilir.**

> **Yaklaşım kanıtlanmıştır.** Bu depodaki Türkçe veri ve talimat-ayarı yaklaşımı, amiral
> modelimiz **[Erk](https://github.com/ecloudtechnology/erk)** için de kullanıldı. Erk
> (Qwen3-14B tabanlı), bağımsız ve kamuya açık **TurkishMMLU** kıyaslamasında **%69,7** ile
> test edilen açık Türkçe modellerin **en iyisi** oldu — Trendyol, Turkish-Gemma ve Kumru dahil.
>
> *Not:* Bu depodaki uçtan uca **sıfırdan eğitim** reçetesi (kendi tokenizer'ını eğit →
> sıfırdan ön-eğit), makul bütçeyle kendi Türkçe modelini eğitmek isteyenler içindir. Erk
> ise güçlü bir açık temel (Qwen3-14B) üzerine Türkçe sürekli ön-eğitim + talimat ayarı ile
> geliştirilmiştir; ikisi farklı senaryolardır.

### Neden Türkçe'ye özel?

Türkçe, sondan eklemeli (agglutinatif) bir dildir: tek bir köke onlarca ek eklenir.
İngilizce için tasarlanmış tokenizer'lar bu yapıyı tanımaz ve kelimeyi gereğinden çok
parçaya böler. Sonuç: **daha yavaş üretim, daha yüksek maliyet, yarı yarıya kısalan
etkin bağlam.**

Bunu ölçtük. 14 tokenizer'ı, bağımsız Türkçe/İngilizce test derlemlerinde (held-out)
karşılaştırdık. Metrik: **fertility** — kelime başına düşen token; düşük olan daha iyidir.

| Tokenizer | Sözlük | Türkçe web ↓ | Türkçe wiki ↓ | Türkçe maliyet |
|:---|---:|---:|---:|---:|
| **erk-64k** *(bu repo)* | 65.536 | **1.51** | **1.71** | **1.00×** |
| cosmosGPT (TR) | 50.258 | 1.56 | 1.75 | 1.03× |
| Kumru-2B (TR) | 50.176 | 1.68 | 1.91 | 1.11× |
| Llama 3.1 | 128.256 | 2.24 | 2.32 | 1.48× |
| GPT-4o (o200k) | 200.019 | 2.26 | 2.35 | 1.50× |
| Qwen 2.5 | 151.665 | 2.61 | 2.73 | 1.72× |
| **GPT-4 (cl100k)** | 100.277 | 3.01 | 3.02 | **1.99×** |
| Kimi K2 | 163.840 | 3.07 | 3.03 | 2.03× |
| Mistral v0.3 | 32.768 | 3.82 | 3.87 | 2.53× |

> **GPT-4'ün tokenizer'ı, aynı Türkçe metni bu reçetenin ürettiği tokenizer'dan tam iki
> kat fazla token'a böler.** Türkçe-native tokenizer, test edilen 14 tokenizer'ın
> (diğer Türkçe modeller dahil) tümünü Türkçe'de geçer.

Tam sonuçlar ve ham JSON: [`results/`](results/).
*Dürüst dezavantaj:* İngilizce'de, İngilizce-odaklı tokenizer'ların gerisindeyiz —
bilinçli bir tercih.

### Reçete — beş aşama, tek kod tabanı

| # | Aşama | Script | Ne yapar |
|:--:|:---|:---|:---|
| 1 | **Tokenizer** | [`corpus/train_tokenizer.py`](corpus/train_tokenizer.py) | Türkçe'ye özel byte-level BPE eğitir |
| 2 | **Veri** | [`corpus/build_shards_tr.py`](corpus/build_shards_tr.py) | Web ölçekli Türkçe derlemi paketler |
| 3 | **Ön-eğitim** | [`scripts/base_train.py`](scripts/base_train.py) | Modeli sıfırdan eğitir |
| 4 | **Talimat ayarı** | [`scripts/chat_sft.py`](scripts/chat_sft.py) | Sohbet yeteneği, alan bilgisi ve kimlik kazandırır |
| 5 | **Sohbet** | [`scripts/chat_cli.py`](scripts/chat_cli.py) | Modelinizle konuşursunuz |

### Hızlı başlangıç

```bash
git clone https://github.com/ecloudtechnology/nanosohbet.git
cd nanosohbet
pip install -e .

# 1) Türkçe tokenizer eğit
python -m corpus.train_tokenizer

# 2) Ön-eğitim verisini hazırla
python -m corpus.build_shards_tr

# 3) Sıfırdan ön-eğitim (çok-GPU'ya ölçeklenir)
torchrun --nproc_per_node=4 -m scripts.base_train -- --depth=20

# 4) Türkçe talimat ayarı
torchrun --nproc_per_node=1 -m scripts.chat_sft

# 5) Modelinle konuş
python -m scripts.chat_cli
```

**Ne kadara mal olur?** Konuşabilen bir Türkçe model ~100 USD, daha yetkin bir model
~300–400 USD bütçeyle eğitilebilir. Gerçek ölçümlere dayalı ayrıntılı hesap:
[**Eğitim Maliyet Rehberi**](MALIYET.md).

Her aşama tek bir A100 düğümünde koşabilir; ön-eğitim çok-GPU'ya ölçeklenir.

### Depo yapısı

```
corpus/     Tokenizer eğitimi, veri hazırlama ve karşılaştırma araçları
scripts/    Ön-eğitim, SFT, değerlendirme ve sohbet
slurm/      HPC (Slurm) iş dosyaları
results/    Tokenizer karşılaştırma sonuçları ve örnek üretimler
```

### Yol haritası

| Faz | İçerik | Durum |
|:--:|:---|:--:|
| 1 | Türkçe BPE tokenizer + 14-tokenizer karşılaştırması | ✅ |
| 2 | Türkçe web derleminde ön-eğitim | ✅ |
| 3 | Türkçe talimat ayarı — sohbet, alan uzmanlığı, kimlik | ✅ |
| 4 | TurkishMMLU değerlendirmesi — **açık Türkçe modellerde 1.** (%69,7) | ✅ |
| 5 | Model ağırlıkları — [🤗 Erk-14B](https://huggingface.co/ecloudtech/Erk-14B) yayında | ✅ |
| 6 | [Eğitim maliyet rehberi](MALIYET.md) — "kendi modelini şu kadara eğit" | ✅ |
| 7 | **Erk-32B** — genişletilmiş ölçekli sürüm | 🔜 çalışmalar başlıyor |

### Katkı

Sorun bildirimleri ve katkılar açıktır. Büyük değişiklikler için önce bir issue
açmanızı rica ederiz. ⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın.

---

## English

### What is nanosohbet?

**nanosohbet** is a complete, open and reproducible **recipe for training a Turkish
language model from scratch.** The goal isn't to ship one model — it's to let **anyone
train their own Turkish LLM on a reasonable budget.**

Anyone who wants to train a model for Turkish today is stuck between giant closed models
and scattered, incomplete resources. nanosohbet fills that gap: from tokenizer training
to pretraining to instruction tuning to a chat interface — **every step, in one codebase,
open and reproducible.**

> **The approach is proven.** We used the Turkish data and instruction-tuning approach in
> this repo for our flagship model **[Erk](https://github.com/ecloudtechnology/erk)** as well.
> Erk (built on Qwen3-14B) ranks **#1 among open Turkish models on the independent, public
> TurkishMMLU benchmark (69.7%)** — ahead of Trendyol, Turkish-Gemma and Kumru.
>
> *Note:* the end-to-end **from-scratch** recipe in this repo (train your own tokenizer →
> pretrain from scratch) is for training your own Turkish model on a budget. Erk instead is
> built on a strong open base (Qwen3-14B) with Turkish continued pretraining + instruction
> tuning — they are different scenarios.

**Why Turkish-specific?** Turkish is agglutinative — dozens of suffixes attach to a single
root. English-centric tokenizers over-split Turkish, meaning slower generation, higher
cost and half the effective context. We measured 14 tokenizers on held-out corpora:
**GPT-4's tokenizer uses 2× more tokens for the same Turkish text** than the tokenizer
this recipe produces, which beats every tokenizer tested — including other Turkish models.
See [`results/`](results/).

**The recipe** is five stages in one codebase: train a Turkish tokenizer → pack a
web-scale Turkish corpus → pretrain from scratch → instruction-tune → chat.

```bash
git clone https://github.com/ecloudtechnology/nanosohbet.git && cd nanosohbet
pip install -e .
python -m corpus.train_tokenizer
python -m corpus.build_shards_tr
torchrun --nproc_per_node=4 -m scripts.base_train -- --depth=20
```

⭐ If you find this useful, please star the repo.

---

<div align="center">

Bir **[eCloud Yazılım Teknolojileri](https://www.e-cloud.web.tr)** projesidir.
<br>
*Yerli zekâ, küresel ölçek.*

</div>
