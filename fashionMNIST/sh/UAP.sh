delta1=0.1
delta2=0.3

# normal, proxy0.5
python src/UAP.py --AK=proxy0.5 --delta=${delta1}  --output_path=output/UAP/proxy0.5 > output/UAP/proxy0.5/delta${delta1}.txt
python src/UAP.py --AK=proxy0.5 --delta=${delta2}  --output_path=output/UAP/proxy0.5 > output/UAP/proxy0.5/delta${delta2}.txt

# adv_train, proxy
python src/UAP.py --AK=robust_white --delta=${delta1}  --output_path=output/UAP/robust_white > output/UAP/robust_white/delta${delta1}.txt
python src/UAP.py --AK=robust_white --delta=${delta2}  --output_path=output/UAP/robust_white > output/UAP/robust_white/delta${delta2}.txt

python src/UAP.py --AK=robust_proxy --delta=${delta1}  --output_path=output/UAP/robust_proxy > output/UAP/robust_proxy/delta${delta1}.txt
python src/UAP.py --AK=robust_proxy --delta=${delta2}  --output_path=output/UAP/robust_proxy > output/UAP/robust_proxy/delta${delta2}.txt

# adv_train, proxy0.5
python src/UAP.py --AK=robust_proxy0.5 --delta=${delta1}  --output_path=output/UAP/robust_proxy0.5 > output/UAP/robust_proxy0.5/delta${delta1}.txt
python src/UAP.py --AK=robust_proxy0.5 --delta=${delta2}  --output_path=output/UAP/robust_proxy0.5 > output/UAP/robust_proxy0.5/delta${delta2}.txt

