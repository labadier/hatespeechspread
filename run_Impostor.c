#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h> 

struct info{
    int index;
    char lang[3];
    char model[15];
};


void * handler( void *arg){

    struct info model = *((struct info *)arg);

    char command[256];
    char coef[8];
    strcpy(coef, "0.");
    for(int i = 10; i <= 90; i+= 5){

        coef[2] = i/10 + '0';
        coef[3] = i%10 + '0';
        
        strcpy(command, "python main.py -l ");
        strcat(command, model.lang);
        strcat(command, " -mode tImpostor -dp data/pan21-author-profiling-training-2021-03-14 -rp ");
        strcat(command, coef);
        strcat(command, " -metric cosine -up random -ecnImp ");
        strcat(command, model.model );
        strcat(command, " -dt $data_test -output logs -interm_layer 64 > res_");
        strcat(command, model.model);
        strcat(command, "_");
        strcat(command, model.lang);
        system(command);
    }

	pthread_exit(NULL);
}

char models[5][15];
void main(){

    pthread_t tid[12];
    strcpy(models[0], "fcnn");
    strcpy(models[1], "gcn");
    strcpy(models[2], "lstm");

    struct info send;
    for(int i = 0; i < 3; i++){

        send.index = i;
        strcpy(send.model, models[i]);
       
        strcpy(send.lang, "EN");
        if(pthread_create(&tid[i*2], NULL, handler, &send) != 0)
            printf("Error\n");
        break;

        strcpy(send.lang, "ES");
        if(pthread_create(&tid[i*2 + 1], NULL, handler, &send) != 0)
            printf("Error\n");
    }
    
    for (int i = 0; i < 1; i++)
       pthread_join(tid[i], NULL);

}