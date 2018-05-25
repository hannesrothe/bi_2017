# bi_2017

Here you'll find some course material for the course Business Intelligence 2017

The folders contain example python code to exemplify the use of certain machine learning techniques, discussed in our course.

Furthermore you will find exercises or exercise data for your free use.

## Installing Anaconda OR Dependencies by hand

I suggest in class to use the Anaconda Suite (https://www.anaconda.com/downloads). It comes with all the necessary dependencies you will need for this class. Additionally it provides you with an IDE (Spyder) as well as a Jupyter Notebook environment. The Anaconda suite can be installed on MAC, Windows and Linux OS.

If you choose to use another development environment, you can install python as well as all dependencies by hand.
**Note:** In this course we use Python 3 and third party libraries.

### Python 3
Install Python 3 on a Mac via [homebrew](https://brew.sh/#install) (A package manager for macOS).

```
# install homebrew
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
# install python 3
brew install python3
```

### Libraries
Following dependencies are required. Install them by using PIP as follows:

```
pip3 install pandas
pip3 install matplotlib 
pip3 install seaborn 
pip3 install sklearn 
```
Adding dependencies with Anaconda is just as simple, instead of using pip, Anaconda uses the conda script. Remember, installing the libraries stated before is not necessary after installing Anaconda. It may be useful for installing further libraries whatsoever:
```
conda install [LIBRARY]
```

## TODOS
- [x] Python 3 mac installation
- [x] Python 3 windows installation
- [ ] Full list of dependencies
