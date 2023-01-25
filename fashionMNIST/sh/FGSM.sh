eps1=0.05
eps2=0.075

# normal, proxy0.5
python src/FGSM.py --AK=proxy0.5 --eps=${eps1}  --output_path=output/FGSM/proxy0.5 > output/FGSM/proxy0.5/eps${eps1}.txt
python src/FGSM.py --AK=proxy0.5 --eps=${eps2}  --output_path=output/FGSM/proxy0.5 > output/FGSM/proxy0.5/eps${eps2}.txt

# adv_train, proxy
python src/FGSM.py --AK=robust_white --eps=${eps1}  --output_path=output/FGSM/robust_white > output/FGSM/robust_white/eps${eps1}.txt
python src/FGSM.py --AK=robust_white --eps=${eps2}  --output_path=output/FGSM/robust_white > output/FGSM/robust_white/eps${eps2}.txt

python src/FGSM.py --AK=robust_proxy --eps=${eps1}  --output_path=output/FGSM/robust_proxy > output/FGSM/robust_proxy/eps${eps1}.txt
python src/FGSM.py --AK=robust_proxy --eps=${eps2}  --output_path=output/FGSM/robust_proxy > output/FGSM/robust_proxy/eps${eps2}.txt

# adv_train, proxy0.5
python src/FGSM.py --AK=robust_proxy0.5 --eps=${eps1}  --output_path=output/FGSM/robust_proxy0.5 > output/FGSM/robust_proxy0.5/eps${eps1}.txt
python src/FGSM.py --AK=robust_proxy0.5 --eps=${eps2}  --output_path=output/FGSM/robust_proxy0.5 > output/FGSM/robust_proxy0.5/eps${eps2}.txt

