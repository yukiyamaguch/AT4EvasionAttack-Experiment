wcon1=0.3
python src/CW.py --AK=white --con=${wcon1} --output_path=output/CW/white > output/CW/white/con${wcon1}.txt
wcon2=0.25
python src/CW.py --AK=white --con=${wcon2} --output_path=output/CW/white > output/CW/white/con${wcon2}.txt
wcon3=0.1
python src/CW.py --AK=white --con=${wcon3} --output_path=output/CW/white > output/CW/white/con${wcon3}.txt
pcon1=0.3
python src/CW.py --AK=proxy --con=${pcon1} --output_path=output/CW/proxy > output/CW/proxy/con${pcon1}.txt
pcon2=0.25
python src/CW.py --AK=proxy --con=${pcon2} --output_path=output/CW/proxy > output/CW/proxy/con${pcon2}.txt
pcon3=0.1
python src/CW.py --AK=proxy --con=${pcon3} --output_path=output/CW/proxy > output/CW/proxy/con${pcon3}.txt
