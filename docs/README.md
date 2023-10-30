# Internal documentation for the XR2Learn Emotion Recognition Enablers

The documentation is managed with an open-source tool [mkdocs](https://www.mkdocs.org/) and the [material](https://squidfunk.github.io/mkdocs-material/) plugin. The documentation is not published to web and can be launched locally.

In order to build the documentation locally:
```
python -m venv venv

source venv/bin/activate

python -m pip install -r requirements.txt

mkdocs serve
```

The last command will outputs the local address, where the documentation will be deployed. Typically, it uses port `8000`.



alternative to build html and self host:
```
mkdocs build
python3 -m http.server 8000 -d .\site\
```
