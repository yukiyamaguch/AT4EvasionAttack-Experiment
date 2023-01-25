gamma1=0.05
gamma2=0.075

# normal, proxy0.5
python src/JSMA.py --AK=proxy0.5 --gamma=${gamma1}  --output_path=output/JSMA/proxy0.5 > output/JSMA/proxy0.5/gamma${gamma1}.txt
python src/JSMA.py --AK=proxy0.5 --gamma=${gamma2}  --output_path=output/JSMA/proxy0.5 > output/JSMA/proxy0.5/gamma${gamma2}.txt

# adv_train, proxy
python src/JSMA.py --AK=robust_white --gamma=${gamma1}  --output_path=output/JSMA/robust_white > output/JSMA/robust_white/gamma${gamma1}.txt
python src/JSMA.py --AK=robust_white --gamma=${gamma2}  --output_path=output/JSMA/robust_white > output/JSMA/robust_white/gamma${gamma2}.txt

python src/JSMA.py --AK=robust_proxy --gamma=${gamma1}  --output_path=output/JSMA/robust_proxy > output/JSMA/robust_proxy/gamma${gamma1}.txt
python src/JSMA.py --AK=robust_proxy --gamma=${gamma2}  --output_path=output/JSMA/robust_proxy > output/JSMA/robust_proxy/gamma${gamma2}.txt

# adv_train, proxy0.5
python src/JSMA.py --AK=robust_proxy0.5 --gamma=${gamma1}  --output_path=output/JSMA/robust_proxy0.5 > output/JSMA/robust_proxy0.5/gamma${gamma1}.txt
python src/JSMA.py --AK=robust_proxy0.5 --gamma=${gamma2}  --output_path=output/JSMA/robust_proxy0.5 > output/JSMA/robust_proxy0.5/gamma${gamma2}.txt

