import csv
from time import time

from sklearn import cross_validation, svm, linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from ClassificationReport import plot_classification_report
from CleaningTweets import tweet_to_words
from ConfusionMatrix import show_confusion_matrix


# train_data.csv contains two columns
# first column is the tweet content (quoted)
# second column is the assigned sentiment (political or not)
def load_file():

    with open('dataset/train_data.csv', 'r', encoding="ISO-8859-1") as file:
        reader = list(csv.reader(file))
        data =[]
        target = []

        for row in reader:
            # skip missing data
            if row[0] and row[1]:
                data.append(tweet_to_words(row[0]))
                target.append(row[1])

        return data,target

# preprocess creates the term frequency matrix for the review data set
def preprocess():
    data,target = load_file()
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)
    return tfidf_data

def evaluate_model(target_true,target_predicted,clf):
    cr = classification_report(target_true,target_predicted)
    plt = plot_classification_report(cr)
    print("Accuracy {:.2%}".format(accuracy_score(target_true, target_predicted)))
    plt.savefig('plots/'+(str(clf).partition("(")[0])+'_CR.pdf', bbox_inches='tight')
    print((str(clf).partition("(")[0]) + " Classification Report saved successfully")


def learn_model(data,target,clf):
    # preparing data for split validation.
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.40,random_state=43)

    clf1 = clf.fit(data_train,target_train)
    pred1 = clf1.predict(data_test)
   # evaluate_model(target_test,pred1,clf1)
    cr = classification_report(target_test, pred1)
    cr_plt = plot_classification_report(cr)
    print("Accuracy {:.2%}".format(accuracy_score(target_test, pred1)))
    cr_plt.savefig('plots/' + (str(clf).partition("(")[0]) + '_CR.pdf', bbox_inches='tight')
    print((str(clf).partition("(")[0]) + " Classification Report saved successfully")
    cr_plt.close()

    cm = confusion_matrix(target_test,pred1)
    cm_plt = show_confusion_matrix(cm, ['NOT', 'POLIT'])
    cm_plt.savefig('plots/'+(str(clf1).partition("(")[0])+'_CM.pdf')
    print((str(clf1).partition("(")[0])+" Confusion Matrix saved successfully")
    cm_plt.close()

def main():
    clf1 = MultinomialNB()
    clf2 = BernoulliNB()
    clf3 = svm.SVC(kernel='linear')
    clf4 = linear_model.LogisticRegression()
    print("Loading and Cleaning the data set....")
    t0 = time()
    data,target = load_file()
    print("done in %0.3fs \n" % (time() - t0))
    print("Creating the Term Frequency Matrix for the data set....")
    t0 = time()
    tf_idf = preprocess()
    print("done in %0.3fs \n" % (time() - t0))
    print("Learning Model : Split Validation, Classifying, Predicting, Classification Report, Confusion Matrix....\n")
    print(str(clf1).partition("(")[0])
    t0 = time()
    learn_model(tf_idf,target,clf1)
    print("done in %0.3fs \n" % (time() - t0))

    print(str(clf2).partition("(")[0])
    t0 = time()
    learn_model(tf_idf, target, clf2)
    print("done in %0.3fs \n" % (time() - t0))

    print(str(clf3).partition("(")[0])
    t0 = time()
    learn_model(tf_idf, target, clf3)
    print("done in %0.3fs \n" % (time() - t0))

    print(str(clf4).partition("(")[0])
    t0 = time()
    learn_model(tf_idf, target, clf4)
    print("done in %0.3fs \n" % (time() - t0))

main()
