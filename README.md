Projeyi https://github.com/zedraxa/ai-agent reposuna taşıdım




Kullanmış olduğum işletim sisteminde(parrotOS) localde openclaw kullanmamın zahmetli olması, autogpt'nin de api'a olan bağlılığı sebebiyle tatilde başladığım ihtiyaca yönelik AI agent projesi. (henüz başarıya ulaşmadı) Githuba yüklenmesini henüz tamamlamadım.
Ollama ile localde bulunan bir model kullanarak çalışıyor.
Ben qwen2.5:7b-instruct kullanıyorum

Kullanım 
cd ~/ai-agent
source venv/bin/activate
python agent.py
kodlarını terminale girdikten sonra agentın açılmasını bekleyin 
daha sonrasında talebinizi girin,eğer talebiniz web araması içeriyorsa talebinize ALLOW_WEB_SEARCH komutunu ekleyin
terminale exit yazarak çıkılabilir

Örnek Senaryo:
 çalıştırmak için:
  cd ~/ai-agent
  source venv/bin/activate
  python agent.py --model qwen2.5:7b-instruct

 prompt:
  PROJECT: wine_quality ALLOW_WEB_SEARCH | UCI Wine Quality datasetini bul; indirilebilir CSV linkini çıkar ve data/raw/ içine indir; src/train.py yaz: pandas+sklearn ile baseline model eğit, accuracy ve ROC-AUC hesapla; results/metrics.json ve report.md üret; sonra training scriptini çalıştır.

  En son halinde bu propmt için göstermiş olduğu metrikler:
  
   
   cat results/metrics.json
{
  "dataset": "UCI Wine Quality (red)",
  "test_size": 0.2,
  "random_state": 42,
  "model": "StandardScaler + LogisticRegression(lbfgs)",
  "accuracy": 0.590625,
  "roc_auc_ovr_macro": 0.7639904835106219
}(
