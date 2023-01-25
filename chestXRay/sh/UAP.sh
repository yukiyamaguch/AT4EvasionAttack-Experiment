wdelta1=0.1
python src/UAP.py --AK=white --eps=1.0 --delta=${wdelta1} --output_path=output/UAP/white > output/UAP/white/delta${wdelta1}.txt

wdelta2=0.3
python src/UAP.py --AK=white --eps=1.0 --delta=${wdelta2} --output_path=output/UAP/white > output/UAP/white/delta${wdelta2}.txt

wdelta3=0.5
python src/UAP.py --AK=white --eps=1.0 --delta=${wdelta3} --output_path=output/UAP/white > output/UAP/white/delta${wdelta3}.txt

pdelta1=0.1
python src/UAP.py --AK=proxy --eps=1.0 --delta=${pdelta1} --output_path=output/UAP/proxy > output/UAP/proxy/delta${pdelta1}.txt

pdelta2=0.3
python src/UAP.py --AK=proxy --eps=1.0 --delta=${pdelta2} --output_path=output/UAP/proxy > output/UAP/proxy/delta${pdelta2}.txt

pdelta3=0.5
python src/UAP.py --AK=proxy --eps=1.0 --delta=${pdelta3} --output_path=output/UAP/proxy > output/UAP/proxy/delta${pdelta3}.txt

