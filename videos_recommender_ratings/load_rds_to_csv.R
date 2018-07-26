library(data.table);

datadir <- '/home/webb/Compass_Recommender/';

print("Loading Events Data...")

content  = readRDS(paste0(datadir,'content.rds'))
events   = readRDS(paste0(datadir,'events.rds'))
learners = readRDS(paste0(datadir,'learners.rds'))

events = merge(events, content,  by='media_id',  all=F)
events = merge(events, learners, by='learner_id',all=F)

dir.create('data')
csv_file_name <- 'data/events.csv'

print(paste("Writing to : ", csv_file_name) ) 
write.csv(events, file='data/events.csv')
