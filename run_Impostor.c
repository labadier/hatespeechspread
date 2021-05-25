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
        strcat(command, " -dt data/pan21-author-profiling-test-without-gold -output logs -interm_layer 64 >> experiments/");
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
    strcpy(models[3], "transformer");
    struct info send[8];
    
    for(int i = 0; i < 4; i++){

        
        send[i*2].index = i;
        send[i*2+1].index = i;
        strcpy(send[i*2].model, models[i]);
        strcpy(send[i*2+1].model, models[i]);
       
        strcpy(send[i*2].lang, "EN");
        if(pthread_create(&tid[i*2], NULL, handler, &send[i*2]) != 0)
            printf("Error\n");

        strcpy(send[i*2+1].lang, "ES");
        if(pthread_create(&tid[i*2 + 1], NULL, handler, &send[i*2+1]) != 0)
            printf("Error\n");
    }
    
    for (int i = 0; i < 8; i++)
       pthread_join(tid[i], NULL);

}