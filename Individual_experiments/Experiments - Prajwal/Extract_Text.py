import json
import csv
import bz2

input_file = open('/Users/prajwalkrishn/Desktop/My_Computer/project - Dsci 601/Models/मुर्खा/मुर्खा_2021-04-28_to_2021-05-03.json')
output_file = open("/Users/prajwalkrishn/Desktop/My_Computer/project - Dsci 601/Models/मुर्खा/मुर्खा_tweets.csv","w")

# data = json.loads(input_file.read())
input_file_decode = json.load(input_file)

result = []
for item in input_file_decode.get('text'):
    print(item)
    my_tweets = {}
    my_tweets['tweets'] = item
    #print(my_tweets)
    result.append(my_tweets)

#back_to_json = json.dump(result)
print(result)

keys = result[0].keys()
writer = csv.DictWriter(output_file,keys)
writer.writeheader()
writer.writerows(result)
input_file.close()
output_file.close()

#list_to_csv = open('salya.csv','w+',newline='')
#
# for key, value in result:
#     write = csv.writer(output_file)
#     write.writerows(result[key][value])

# with output_file:
#     write = csv.writer(output_file)
#     write.writerows(result[0]['tweets'])

# input_file.close()
# output_file.close()


# def bar(**kwargs):
#     for a in kwargs:
#         print(a, kwargs[a])
#
# bar(name='one', age=27)
#
#
# def hello_world(**kwargs):
#     for key, value in kwargs.items():
#         print("{0} = {1}".format(key, value))
#
# hello_world(hello = "world")