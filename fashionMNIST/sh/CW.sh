con1=1.0
con2=0.2

python src/CW.py --AK=proxy --con=${con2}  --output_path=output/CW/proxy > output/CW/proxy/con${con2}.txt
# normal, proxy0.5
#python src/CW.py --AK=proxy0.5 --con=${con1}  --output_path=output/CW/proxy0.5 > output/CW/proxy0.5/con${con1}.txt
python src/CW.py --AK=proxy0.5 --con=${con2}  --output_path=output/CW/proxy0.5 > output/CW/proxy0.5/con${con2}.txt
#
## adv_train, proxy
#python src/CW.py --AK=robust_white --con=${con1}  --output_path=output/CW/robust_white > output/CW/robust_white/con${con1}.txt
python src/CW.py --AK=robust_white --con=${con2}  --output_path=output/CW/robust_white > output/CW/robust_white/con${con2}.txt
#
#python src/CW.py --AK=robust_proxy --con=${con1}  --output_path=output/CW/robust_proxy > output/CW/robust_proxy/con${con1}.txt
python src/CW.py --AK=robust_proxy --con=${con2}  --output_path=output/CW/robust_proxy > output/CW/robust_proxy/con${con2}.txt
#
## adv_train, proxy0.5
#python src/CW.py --AK=robust_proxy0.5 --con=${con1}  --output_path=output/CW/robust_proxy0.5 > output/CW/robust_proxy0.5/con${con1}.txt
python src/CW.py --AK=robust_proxy0.5 --con=${con2}  --output_path=output/CW/robust_proxy0.5 > output/CW/robust_proxy0.5/con${con2}.txt

