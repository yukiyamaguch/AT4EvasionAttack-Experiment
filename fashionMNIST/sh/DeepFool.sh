eps1=0.5
eps2=0.25

# normal, proxy0.5
python src/DeepFool.py --AK=proxy0.5 --eps=${eps1}  --output_path=output/DeepFool/proxy0.5 > output/DeepFool/proxy0.5/eps${eps1}.txt
python src/DeepFool.py --AK=proxy0.5 --eps=${eps2}  --output_path=output/DeepFool/proxy0.5 > output/DeepFool/proxy0.5/eps${eps2}.txt

# adv_train, proxy
python src/DeepFool.py --AK=robust_white --eps=${eps1}  --output_path=output/DeepFool/robust_white > output/DeepFool/robust_white/eps${eps1}.txt
python src/DeepFool.py --AK=robust_white --eps=${eps2}  --output_path=output/DeepFool/robust_white > output/DeepFool/robust_white/eps${eps2}.txt

python src/DeepFool.py --AK=robust_proxy --eps=${eps1}  --output_path=output/DeepFool/robust_proxy > output/DeepFool/robust_proxy/eps${eps1}.txt
python src/DeepFool.py --AK=robust_proxy --eps=${eps2}  --output_path=output/DeepFool/robust_proxy > output/DeepFool/robust_proxy/eps${eps2}.txt

# adv_train, proxy0.5
python src/DeepFool.py --AK=robust_proxy0.5 --eps=${eps1}  --output_path=output/DeepFool/robust_proxy0.5 > output/DeepFool/robust_proxy0.5/eps${eps1}.txt
python src/DeepFool.py --AK=robust_proxy0.5 --eps=${eps2}  --output_path=output/DeepFool/robust_proxy0.5 > output/DeepFool/robust_proxy0.5/eps${eps2}.txt


