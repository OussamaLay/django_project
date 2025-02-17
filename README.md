# neuro

cd neuro

python -m venv .djangoenv
python.exe -m pip install --upgrade pip

cd .djangoenv\Scripts\
./activate
cd ../..
pip install -r requirements.txt

set TF_ENABLE_ONEDNN_OPTS=0

python manage.py migrate

python manage.py runserver



