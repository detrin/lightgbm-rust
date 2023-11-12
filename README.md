## Boosting with rust

## How to run
To generate data
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

```bash
brew install cmake libomp libiconv
brew unlink libomp
brew install libomp.rb
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib -L/opt/homebrew/opt/libiconv/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include -I/opt/homebrew/opt/libiconv/include"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export LIBRARY_PATH=$LIBRARY_PATH:$(brew --prefix)/lib:$(brew --prefix)/opt/libiconv/lib:$(brew --prefix)/opt/libomp/lib
```