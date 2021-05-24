#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h> 

struct info{

    int index;
    char lang[3];
    char model[15];
}

void * handler( void *arg){

    char model[15];
    struct index model = *((struct info *)arg);

    char command[70];
    char coef[8];
    for(int i = 1; i <= 90; i+= 5){
        strcpy(coef, "0.");
        strcat(coef, itoa(i));
        strcpy(command, "python main.py -l ");
        strcat(command, lang);
        strcat(command, " -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp ");
        strcat(command, coef);
        strcat(command, " -metric cosine -up random -ecnImp transformer -dt $data_test -output logs -interm_layer 64");
        system(command);
    }

	pthread_exit(NULL);
}


void main(){

    pthread_t tid[12];
    char *models[] = {"fcnn", "gcn", "lstm"}

    struct info send;
    for(int i = 0; i < 3; i++){

        send.index = i;
        strcpy(send.model, models[i]);
        strcpy(send.lang, "EN");
        if(pthread_create(&tid[i*2], NULL, handler, &send) != 0)
            printf("Error\n");

        strcpy(send.lang, "ES");
        if(pthread_create(&tid[i*2 + 1], NULL, handler, &send) != 0)
            printf("Error\n");
    }

    for (int i = 0; i < 6; i++)
       pthread_join(tid[i], NULL);

}