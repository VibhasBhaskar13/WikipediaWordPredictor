WikiWordPredictor is a project developed in python. First, there is a scraper (scraper.py) that uses sockets and spiders wikipedia (following guidelines like robot.txt). It stores its info in the sqlite database.
Then, the python code uses that database to train a simple Neural Network using pytorch.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Scraper (scraper.py):

-Started with a starting point (a wikipedia page)

-Asks user for how many pages it should iterate through.

-Goes thorugh page, stores all links and adds them to the queue which is saved in a pickle file after spidering ends so that user can pick up from ending point in next run.

-All the data from the pages is stored in the sqlite database

----------------------------------------------------------------------------------------------------------

Neural Network (main.py):

-Uses pytorch

-Takes in data from the sqlite database and formats it so that only that actual words of the article remain

-The data is used in creating the samples and the labels

-It uses the GoogleNews word2Vec (which can be found on Kaggle at this link: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300/

-It then filters throught the 3,000,000 words given by that dataset so that only the most common 100,000 remain for effiency purposes

-Then, after formatting, everything is sent to the neural network which trains (starting at a learning rate of 1%, which is a little high for Adam)

-Learning rate goes down by using the scheduler
