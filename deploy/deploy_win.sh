# This is your one-stop shop for installing the server on a new machine.
# If you run into a missing dependency that prevents this from working on a
# brand-new machine, make sure to add it here!

# :: Install Python 3.8-venv
python -m venv venv

# :: Install dependencies
pip install wheel==0.37.1
pip install cython
pip install -r requirements.txt

# :: Clone and install peewee
git clone https://github.com/coleifer/peewee.git
cd peewee
python setup.py install
python setup.py build_ext -i
cd ..

# :: Navigate to the deploy/systemd directory
cd deploysystemd

# :: Run deploy script
bash deploy_win.sh
