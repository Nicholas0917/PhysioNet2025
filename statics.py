from helper_code import *
from helper_code import find_records, get_age, get_sex, load_label

data_folder = '/mnt/scratch/wmqn2362/PhysioNet25/data/'
records = find_records(data_folder)
num_records = len(records)
for i in range(num_records):
    records[i] = os.path.join(data_folder, records[i])

# Load the labels
labels = []
ages = []
sexs = []

for record in records:
    label = load_label(record)
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)
    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1
    labels.append(label)
    ages.append(age)
    sexs.append(one_hot_encoding_sex)

# Calculate the mean, standard deviation, maximum, and minimum of ages
mean_age = np.mean(ages)
std_age = np.std(ages)
max_age = np.max(ages)
min_age = np.min(ages)

# Print the results
print(f'Mean age: {mean_age:.2f}')
print(f'Standard deviation of age: {std_age:.2f}')
print(f'Maximum age: {max_age:.2f}')
print(f'Minimum age: {min_age:.2f}')

# Count the number of each sex
female_count = sum(sex[0] for sex in sexs)
male_count = sum(sex[1] for sex in sexs)
unknown_count = sum(sex[2] for sex in sexs)

# Calculate the proportions
female_proportion = female_count / len(sexs)
male_proportion = male_count / len(sexs)
unknown_proportion = unknown_count / len(sexs)

# Print the results
print(f'Number of females: {female_count}')
print(f'Number of males: {male_count}')
print(f'Proportion of females: {female_proportion:.2f}')
print(f'Proportion of males: {male_proportion:.2f}')

# Count the number of positive and negative examples
positive_count = sum(labels)
negative_count = len(labels) - positive_count

# Calculate the proportions
positive_proportion = positive_count / len(labels)
negative_proportion = negative_count / len(labels)

# Print the results
print(f'Number of positive examples: {positive_count}')
print(f'Number of negative examples: {negative_count}')
print(f'Proportion of positive examples: {positive_proportion:.2f}')
print(f'Proportion of negative examples: {negative_proportion:.2f}')

