weps1=15
python src/UAP.py --AK=white --eps={weps1} --output_path=output/UAP/white > output/UAP/white/eps${weps1}.txt

weps2=20
python src/UAP.py --AK=white --eps={weps2} --output_path=output/UAP/white > output/UAP/white/eps${weps2}.txt

peps1=15
python src/UAP.py --AK=proxy --eps={peps1} --output_path=output/UAP/proxy > output/UAP/proxy/eps${peps1}.txt

peps2=20
python src/UAP.py --AK=proxy --eps={peps2} --output_path=output/UAP/proxy > output/UAP/proxy/eps${peps2}.txt


