eps1=0.05
eps2=0.075

# normal, proxy0.5
python src/PGD.py --AK=proxy0.5 --eps=${eps1}  --output_path=output/PGD/proxy0.5 > output/PGD/proxy0.5/eps${eps1}.txt
python src/PGD.py --AK=proxy0.5 --eps=${eps2}  --output_path=output/PGD/proxy0.5 > output/PGD/proxy0.5/eps${eps2}.txt

# adv_train, proxy
python src/PGD.py --AK=robust_white --eps=${eps1}  --output_path=output/PGD/robust_white > output/PGD/robust_white/eps${eps1}.txt
python src/PGD.py --AK=robust_white --eps=${eps2}  --output_path=output/PGD/robust_white > output/PGD/robust_white/eps${eps2}.txt

python src/PGD.py --AK=robust_proxy --eps=${eps1}  --output_path=output/PGD/robust_proxy > output/PGD/robust_proxy/eps${eps1}.txt
python src/PGD.py --AK=robust_proxy --eps=${eps2}  --output_path=output/PGD/robust_proxy > output/PGD/robust_proxy/eps${eps2}.txt

# adv_train, proxy0.5
python src/PGD.py --AK=robust_proxy0.5 --eps=${eps1}  --output_path=output/PGD/robust_proxy0.5 > output/PGD/robust_proxy0.5/eps${eps1}.txt
python src/PGD.py --AK=robust_proxy0.5 --eps=${eps2}  --output_path=output/PGD/robust_proxy0.5 > output/PGD/robust_proxy0.5/eps${eps2}.txt

