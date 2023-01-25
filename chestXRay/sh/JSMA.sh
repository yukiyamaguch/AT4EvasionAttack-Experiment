wgamma1=0.05
python src/JSMA.py --AK=white --gamma=${wgamma1} --output_path=output/JSMA/white > output/JSMA/white/gamma${wgamma1}.txt

wgamma2=0.075
python src/JSMA.py --AK=white --gamma=${wgamma2} --output_path=output/JSMA/white > output/JSMA/white/gamma${wgamma2}.txt

pgamma1=0.05
python src/JSMA.py --AK=proxy --gamma=${pgamma1} --output_path=output/JSMA/proxy > output/JSMA/proxy/gamma${pgamma1}.txt

pgamma2=0.075
python src/JSMA.py --AK=proxy --gamma=${pgamma2} --output_path=output/JSMA/proxy > output/JSMA/proxy/gamma${pgamma2}.txt

