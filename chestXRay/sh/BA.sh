query1=1000
query2=2000

# adv_train, proxy
python src/BA.py --AK=white --query=${query1}  --output_path=. > query${query1}.txt
#python src/BA.py --AK=white --query=${query1}  --output_path=output/BA/white > output/BA/white/query${query1}.txt
#python src/BA.py --AK=white --query=${query2}  --output_path=output/BA/white > output/BA/white/query${query2}.txt

