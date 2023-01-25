query1=300
query2=200


# adv_train, proxy
python src/GeoDA.py --AK=white --query=${query1}  --output_path=output/GeoDA/white > output/GeoDA/white/query${query1}.txt
python src/GeoDA.py --AK=white --query=${query2}  --output_path=output/GeoDA/white > output/GeoDA/white/query${query2}.txt

