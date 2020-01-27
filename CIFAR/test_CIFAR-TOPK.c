#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>
#include <sys/time.h>


#define BIG_HAMM 256
#define SUPER_BIG_HAMM 32000
#define N_GLOBAL_RADS 1
#define N_GLOBAL_KAS 10

/*
* matrix struct
* n_rows: Number of rows on matrix mat.
* n_cols: Number of columns on matrix mat.
* mat: List of elements on matrix.
* idx: List of idx of 
*/

int *global_array;

int cmp(const void *a, const void *b){
    int ia = *(int *)a;
    int ib = *(int *)b;
    return global_array[ia] < global_array[ib] ? -1 : global_array[ia] > global_array[ib];
}



typedef struct _matrix
{
    unsigned int n_rows;
    unsigned int n_cols;
    uint_fast8_t **mat;

} matrix;


/*
* list of idx of knn
*/
typedef struct _knn_matrix
{
    unsigned int n_rows;
    unsigned int n_cols;
    int **knn_idx;
} knn_matrix;

/*
* list of idx of knn
*/
typedef struct _groundtruth_matrix
{
    unsigned int n_rows;
    unsigned int n_cols;
    int **mat;
} groundtruth_matrix;


typedef struct _hash_ensemble
{
    unsigned int n_tables;
    matrix **tables;

} hash_ensemble;


/*
* Error message and stop programm
*/
void error(char *message)
{
    printf("Error: %s\n", message);
    exit(EXIT_FAILURE);
}

/*
* Check errors
*/
void check_errors(int argc, char *argv[])
{
    if (argc < 1)
        error("Faltan Parámetros\n");
}

/*
* Create matrix with 0's on each row/column.
* Set number of rows and col.
* set 0's on idx list.
*/
matrix *init_matrix(matrix *mtrx, int row, int col)
{
    int i;
    mtrx->mat = calloc(row+1, sizeof(uint_fast8_t*));
    for(i = 0;i<row;i++) mtrx->mat[i] = calloc(col+1, sizeof(uint_fast8_t));
    mtrx->n_rows = row;
    mtrx->n_cols = col;
    return mtrx;
}

/*
* Create matrix with 0's on each row/column.
* Set number of rows and col.
* set 0's on idx list.
*/
knn_matrix *init_knn_matrix(knn_matrix *knn, int row, int col)
{
    int i;
    knn->knn_idx = calloc(row+1, sizeof(int*));
    for(i = 0;i<row;i++) knn->knn_idx[i] = calloc(col+1, sizeof(int));
    knn->n_rows = row;
    knn->n_cols = col;
    return knn;
}

/*
* show matrix values
*/
void show_matrix(matrix *mtrx)
{
    int i,j;
    for(i = 0;i<mtrx->n_rows;i++)
    {
        for(j = 0;j<mtrx->n_cols;j++)
            printf("% " PRIuFAST8, mtrx->mat[i][j] );
        printf("\n");
    }
}

/*
* get 2**n
*/
int pow_2(int n)
{
    int i,a=2;
    if(n == 0)
        return 1;
    for(i=1;i<n;i++)
        a*=2;
    return a;
}

/*
* fill matrix
* Initiate matrix with 0's, then fill the matrix with a decimal representation of each 8 binary bits
*/
matrix* fill_matrix(matrix *mtrx, char* data)
{
    int row,col;
    FILE* ffile;

    ffile=fopen(data,"r");
    if (ffile==NULL)
    {
        char message[23]="Error abriendo archivo ";
        error(strcat(message,data));
    }
    fscanf(ffile,"%d %d\n",&row,&col);
    mtrx = init_matrix(mtrx,row,col/8);

    /* Read each element from data.dat  */
    int i,j,l,byte_flag;
    for(i = 0;i<mtrx->n_rows;i++)
    {
        for(j = 0;j<mtrx->n_cols;j++)
        {
            for(l=0;l<8;l++)
            {
                fscanf(ffile,"%d ", &byte_flag);
                if (byte_flag)
                    mtrx->mat[i][j] += pow_2(7-l);
            }
        }
    }

    fclose(ffile);

    return mtrx;
}

groundtruth_matrix* fill_groundtruth_matrix(groundtruth_matrix *mtrx, char* filename, int row, int col)
{
   
    FILE* ffile;
    int i,j,value;

    printf("ATTEMPTING TO READ %d ROWS AND %d COLUMNS FROM FILE %s\n",row,col,filename);
    ffile=fopen(filename,"r");

    if (ffile==NULL)
    {
        char message[23]="Error abriendo archivo ";
        error(strcat(message,filename));
    }

    mtrx->mat = calloc(row+1, sizeof(int*));
    for(i = 0;i<row;i++) mtrx->mat[i] = calloc(col+1, sizeof(int));
    mtrx->n_rows = row;
    mtrx->n_cols = col;

    /* Read each element from data.dat  */

    for(i = 0;i<mtrx->n_rows;i++)
    {
        for(j = 0;j<mtrx->n_cols;j++)
        {

            fscanf(ffile,"%d, ", &value);
            mtrx->mat[i][j] = value;
        }

    }

    fclose(ffile);
/*
    printf("ELEMENTS FIRST ROW\n");
    for(j = 0;j<mtrx->n_cols;j++)
         printf("%d, ",mtrx->mat[0][j]);
    printf("\n");
    for(j = 0;j<mtrx->n_cols;j++)
         printf("%d, ",mtrx->mat[1][j]);
    printf("\n");
*/
    return mtrx;
}


/*
* get hamming distance with xor operator and list of same len
*/
int hamming_distance(uint_fast8_t *hash1, uint_fast8_t *hash2, int n)
{
    uint bit_in_char[] = {  0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
                            3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
                            3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
                            2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
                            3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
                            5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
                            2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
                            4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
                            4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
                            5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
                            5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,};
    int dis=0,j,pos;
    for(j=0;j<n;j++)
    {
        pos=hash1[j] ^ hash2[j];
        dis+=bit_in_char[pos];
    }
    return dis;
}

/*
* get a list of distance of all element in mtrx with hash1
*/
int* all_distances(uint_fast8_t *hash, matrix *mtrx, int n)
{
    int i;
    int *d = calloc(mtrx->n_rows+1, sizeof(int));
    for(i=0;i<mtrx->n_rows;i++)
        d[i]=hamming_distance(hash, mtrx->mat[i],n);
    return d;
}

/*
* get a list of distance of all element in mtrx with hash1
*/
int compute_distances(int* distances, uint_fast8_t *hash, matrix *mtrx, int n)
{
    int i;

    for(i=0;i<mtrx->n_rows;i++)
        distances[i]=hamming_distance(hash, mtrx->mat[i],n);
    
    return 1;
}


int update_ensemble_distances(int* distances_one_table, int* distances_ensemble, int n_database)
{
    int i;

    for(i=0;i<n_database;i++)
        distances_ensemble[i]=distances_ensemble[i]+distances_one_table[i];
    
    return 1;
}


/*
* get a list of distance of all element in mtrx with hash1
*/

/*
distance_matrix* all_pairwise_distances(distance_matrix *dmatrix, matrix* querys, matrix* recover)
{
  
    int n_queries = querys->n_rows;
    int n_database = recover->n_rows;
    dmatrix->n_rows = n_queries;
    dmatrix->n_cols = n_database;
    int nbits = querys->n_cols;
    int i,j;

    dmatrix->mat = calloc(n_queries+1, sizeof(int*));
    for(i=0;i<n_queries;i++){
        dmatrix->mat[i] = calloc(n_database+1, sizeof(int));
        for(j=0;j<n_database;j++){
            dmatrix->mat[i][j] = hamming_distance(querys->mat[i], recover->mat[j], nbits);
        }
        printf("END QUERY %d. FIRST HAMMING= %d ...\n",i,dmatrix->mat[i][0]);
    }

    return dmatrix;
}

*/

/*
* get pos of min value on list d.
*/
int* get_idxs(int *d, int n_dist, int k, int *knn_idx)
{
    int i,j1,j2,j;
    int *knn_dist = calloc(k+1,sizeof(int));
    for(j=0;j<k;j++)
        knn_dist[j] = 128;

    for(i=0;i<n_dist;i++)    
        for(j1=0;j1<k;j1++)
            if(d[i]<=knn_dist[j1])
            {
                for(j2=k;j2>j1;j2--)
                {
                    knn_dist[j2-1] = knn_dist[j2-2];
                    knn_idx[j2-1] = knn_idx[j2-2];
                }
                knn_dist[j1] = d[i];
                knn_idx[j1] = i;
                break;
            }

    free(knn_dist);
    return knn_idx;
}

int* get_idxs_and_distances(int *d, int n_dist, int k, int *knn_idx, int *knn_dist)
{
    int i,j1,j2,j;
   
    for(j=0;j<k;j++)
        knn_dist[j] = SUPER_BIG_HAMM;

    for(i=0;i<n_dist;i++)    
        for(j1=0;j1<k;j1++)
            if(d[i]<=knn_dist[j1])
            {
                for(j2=k;j2>j1;j2--)
                {
                    knn_dist[j2-1] = knn_dist[j2-2];
                    knn_idx[j2-1] = knn_idx[j2-2];
                }
                knn_dist[j1] = d[i];
                knn_idx[j1] = i;
                break;
            }

    return knn_idx;
}

/*
* find_knn
* Get knn from querys.
*/
void find_knn(matrix* querys, matrix* recover, knn_matrix* knn)
{
    int *d = NULL;
    int i;

    for(i=0;i<querys->n_rows;i++)
    {
        printf("%d/%d\r",i+1,querys->n_rows);
        fflush(stdout);

        d = all_distances(querys->mat[i], recover,querys->n_cols);
        knn->knn_idx[i]=get_idxs(d,recover->n_rows, knn->n_cols,knn->knn_idx[i]);

        free(d);
    }
    printf("\n");
}

/*
* Save knn on k-nn.dat file
*/
void save_knn(matrix* mtrx, knn_matrix *knn, int k)
{
    char *dirname;
    size_t sz;
    sz = snprintf(NULL, 0, "%d-nn.dat",k);
    dirname = (char *)malloc(sz + 1);
    snprintf(dirname, sz+1, "%d-nn.dat",k);
    FILE* ffile;
    ffile =fopen(dirname,"w+");
//    ffile =fopen("knn.dat","w+");
    

    fprintf(ffile, "%d\n",k);
    int i,j;
    for(i=0;i<knn->n_rows;i++)
        for(j=0;j<knn->n_cols;j++)
        {
            printf("%d\n", knn->knn_idx[i][j]);
            fprintf(ffile, "\n");
        }
    fclose(ffile);
    free(dirname);
}


/*
* free memory allocated to matrix
*/
void free_knn_matrix(knn_matrix *knn)
{
    int i;
    for(i=knn->n_rows;i>=0;i--) free(knn->knn_idx[i]);
    free(knn->knn_idx);
    free(knn);
}

/*
* free memory allocated to matrix
*/

/*
void free_distance_matrix(distance_matrix *dmatrix)
{
    int i;
    for(i=dmatrix->n_rows;i>=0;i--) free(dmatrix->mat[i]);
    free(dmatrix->mat);
    free(dmatrix);
}

*/

/*
* free memory allocated to matrix
*/
void free_matrix(matrix *mtrx)
{
    int i;
    for(i=mtrx->n_rows;i>=0;i--) free(mtrx->mat[i]);
    free(mtrx->mat);
    free(mtrx);
}

void free_ensemble(hash_ensemble* ensemble)
{
    int i;
    for(i=0;i<ensemble->n_tables;i++) 
        free_matrix(ensemble->tables[i]);
    free(ensemble);
}

void evaluate_hashing_global_ball(hash_ensemble* hashed_database, hash_ensemble* hashed_queries, int n_database, int n_queries, int max_tables, groundtruth_matrix *groundtruth, char* filename_results, float min_r, float max_r, float* table_weights){


    int rad_local_search = 0;

    int n_global_rads = N_GLOBAL_RADS; 
    //float rads_global_search[N_GLOBAL_RADS] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
    float rads_global_search[N_GLOBAL_RADS] = {1.0};

    int idx;
    float delta_rad = (max_r-min_r)/(1.0*N_GLOBAL_RADS-1);
    rads_global_search[0] = min_r;
    for(idx=1;idx<N_GLOBAL_KAS;idx++)
        rads_global_search[idx] = rads_global_search[idx-1] + delta_rad;
    
    float precision_list[N_GLOBAL_RADS] = {0.0};
    float recall_list[N_GLOBAL_RADS] = {0.0};
    int hit_counters[N_GLOBAL_RADS] = {0};
    int predicted_counters[N_GLOBAL_RADS] = {0};

    int n_candidates[N_GLOBAL_RADS] = {0};
    int n_empty[N_GLOBAL_RADS] = {0};

    int i,j,k,m,r;
    int n_bits;

    int *distances_query_dabatase_one_table = calloc(n_database+1, sizeof(int));
    float *distances_query_dabatase_ensemble = calloc(n_database+1, sizeof(float));
    int *distances_query_dabatase_minimum = calloc(n_database+1, sizeof(int));
    int* boolean_groundtruth = calloc(n_database+1, sizeof(int));
   
    int topK = 1000;

    int *indices_topK = calloc(topK, sizeof(int));
    float *distances_topK = calloc(topK, sizeof(float));

    float ensemble_distance;
    float one_precision;
    float one_recall;

    float global_av_distance_similar = 0.0;
    float global_av_distance_disimilar = 0.0;
    float global_min_similar = 0.0;
    float global_min_dissimilar = 0.0;
    float av_raw_sum = 0.0;

    float av_hits_topK[9] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    float av_prec_topK[9] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

    int values_topK[9] = {1,5,10,20,50,100,200,500,1000};

    printf("PROCESSING QUERIES WITH %d TABLES ...\n",max_tables);

    double average_retrieval_time = 0.0;
    double total_retrieval_time = 0.0;

    for(i=0;i<n_queries;i++){
        //printf("PROCESSING QUERY %d WITH %d TABLES ...\n",i,max_tables);
        struct timeval start, end;
        gettimeofday(&start, NULL);

        for(k=0;k<n_database;k++){
            distances_query_dabatase_ensemble[k]=0.0;
            distances_query_dabatase_minimum[k]=BIG_HAMM;
            boolean_groundtruth[k]=0;
        }

        int ndesired=0;
        for(m=0;m<groundtruth->n_cols;m++)
            if(groundtruth->mat[i][m]>=0){
                boolean_groundtruth[groundtruth->mat[i][m]]=1;
                ndesired +=1;
            }

        for(j=0;j<max_tables;j++){
            n_bits = hashed_queries->tables[j]->n_cols;
            float weight = table_weights[j];
            for(k=0;k<n_database;k++){
                distances_query_dabatase_one_table[k]=hamming_distance(hashed_queries->tables[j]->mat[i], hashed_database->tables[j]->mat[k],n_bits);       
                float new_distance = (float)distances_query_dabatase_one_table[k];
                //printf("NEW DISTANCE TABLE: %f\n",new_distance);

                new_distance = weight*new_distance;
                distances_query_dabatase_ensemble[k]=distances_query_dabatase_ensemble[k]+new_distance;

                //printf("NEW DISTANCE ENSEMBLE: %f\n",distances_query_dabatase_ensemble[k]);

                if(distances_query_dabatase_minimum[k]>distances_query_dabatase_one_table[k])
                    distances_query_dabatase_minimum[k]=distances_query_dabatase_one_table[k];
            }

        }
        
        printf("\n");

        for(r=0;r<n_global_rads;r++){
            hit_counters[r]=0;
            predicted_counters[r]=0;
            n_candidates[r] = 0;
        }


        float av_distance_similar = 0.0;
        float av_distance_disimilar = 0.0;
        float min_similar = 0.0;
        float min_dissimilar = 0.0;
        float raw_sum = 0.0;

        int nsimilar = 0;
        int ndisimilar = 0;


        int m,n;

        for(m=0; m<topK; m++){

            indices_topK[m] = -1;
            distances_topK[m] = 300.0*300.0 + 1;
                    
        }

        for(k=0;k<n_database;k++){
            //printf("QUERY %d, MIN HAMM DISTANCE TO DATABASE POINT %d IS %d, ENS.DISTANCE IS %d...\n",i,k,distances_query_dabatase_minimum[k],distances_query_dabatase_ensemble[k]);
            ensemble_distance = distances_query_dabatase_ensemble[k];
            //if( (i%100==0) && (k%1000==0))
                //printf("ENSEMBLE DISTANCE QUERY %d to ITEM %d IS %f I.E. %d BITS. MIN IS %d. IS THE POINT RELEVANT=%d ...\n",i,k,ensemble_distance,distances_query_dabatase_ensemble[k],distances_query_dabatase_minimum[k],boolean_groundtruth[k]);
            
            if(distances_query_dabatase_minimum[k] <= rad_local_search){
                if(ensemble_distance < distances_topK[topK-1]){
                    for(m=0; m<topK; m++){

                        if(distances_topK[m]>ensemble_distance){

                            for(n=topK-1;n>m;n--){
                                distances_topK[n]=distances_topK[n-1];
                                indices_topK[n]=indices_topK[n-1];
                            }

                            distances_topK[m] = ensemble_distance;
                            indices_topK[m] = k;

                            break;
                        }

                    }
                }
            }


            if(boolean_groundtruth[k] > 0){ //relevant result
                av_distance_similar += ((float) distances_query_dabatase_ensemble[k]);
                min_similar += ((float) distances_query_dabatase_minimum[k]);
                raw_sum +=  ((float) distances_query_dabatase_minimum[k]);
                nsimilar++;
            } else {
                av_distance_disimilar += ((float) distances_query_dabatase_ensemble[k]);
                min_dissimilar += ((float) distances_query_dabatase_minimum[k]);
                raw_sum -=  ((float) distances_query_dabatase_minimum[k]);
                ndisimilar++;
            }

            if(distances_query_dabatase_minimum[k] <= rad_local_search){
                
                for(r=0;r<n_global_rads;r++){
                    if(ensemble_distance <= rads_global_search[r]){
                        //selected globally
                        predicted_counters[r]=predicted_counters[r]+1;
                        if(boolean_groundtruth[k]>0) 
                            hit_counters[r]=hit_counters[r]+1;
                        n_candidates[r] = n_candidates[r]+1;
                    }
                }

            }

         }

        int count_K_predicted = 0;

        for(m=0; m<topK; m++){

            if(indices_topK[m] > 0)
                count_K_predicted++;
                    
        }

        int t;
        printf("PREDICTED: %d\n",count_K_predicted);
        for(t=0;t<9;t++){
            
            int this_topK = values_topK[t];
            int this_hits_topK = 0;

            for(m=0; m<this_topK; m++){

                //printf("TOP%d, %d, DIST=%f\n",m,indices_topK[m],distances_topK[m]);
                if(boolean_groundtruth[indices_topK[m]]>0) 
                    this_hits_topK += 1;

            }

            //printf("HITS FOR FIRST %d = %d\n",this_topK,this_hits_topK);
            av_hits_topK[t] += ((float) this_hits_topK)/((float) n_queries);

            int den_for_prec = count_K_predicted;
            if(den_for_prec>this_topK)
                den_for_prec = this_topK;

            if(count_K_predicted>0)
                av_prec_topK[t] += ((float) this_hits_topK)/(((float) n_queries)*((float) den_for_prec));
            else
                av_prec_topK[t] += 1;
        }

        av_distance_similar = av_distance_similar/((float)nsimilar);
        av_distance_disimilar = av_distance_disimilar/((float)ndisimilar);
        av_distance_similar = av_distance_similar/((float)max_tables);
        av_distance_disimilar = av_distance_disimilar/((float)max_tables);
        av_raw_sum += raw_sum/((float)n_database);
        min_similar = min_similar/((float)nsimilar);
        min_dissimilar = min_dissimilar/((float)ndisimilar);
        min_similar = min_similar/((float)max_tables);
        min_dissimilar = min_dissimilar/((float)max_tables);

        //printf("QUERY %d, NSIM=%d, NDISS=%d, AV_DIST_SIM=%f, AV_DIST_DISSIM=%f\n",i,nsimilar,ndisimilar,av_distance_similar,av_distance_disimilar);
        //printf("..................... MIN_DIST_SIM=%f, MIN_DIST_DISSIM=%f\n",min_similar,min_dissimilar);
   
        global_av_distance_similar +=  av_distance_similar;
        global_av_distance_disimilar += av_distance_disimilar;
        global_min_similar += min_similar;
        global_min_dissimilar += min_dissimilar;

        for(r=0;r<n_global_rads;r++){
            
            if(predicted_counters[r] > 0)
                one_precision=((float)hit_counters[r])/((float)predicted_counters[r]);
            else
                one_precision=1.0;

            one_recall=((float)hit_counters[r])/((float)ndesired);
            precision_list[r]=precision_list[r]+(one_precision/(float)n_queries);
            recall_list[r]=recall_list[r]+(one_recall/(float)n_queries);
            //printf("@@@ LOCAL RADIUS %d: %d VALID CANDIDATES FOR QUERY %d\n",rad_local_search,n_candidates, i);
            //printf("@@@ GLOBAL RADIUS %f: HITS %d, PREDICTED %d, DESIRED %d, PRECISION %f, RECALL %f\n",rads_global_search[r],hit_counters[r],predicted_counters[r],ndesired,one_precision,one_recall);
      
        }

        for(r=0;r<n_global_rads;r++){
            if(n_candidates[r] == 0){
                printf("EMPTY LIST FOR QUERY %d, RADIUS %f\n",i,rads_global_search[r]);
                n_empty[r]=n_empty[r]+1;
            }
        }

        gettimeofday(&end, NULL);
        long time_secs =  end.tv_sec - start.tv_sec;
        double time_secs_all = ((double)((time_secs*1000000+end.tv_usec) - start.tv_usec))/1000000.0;

        printf("Time in seconds: %f \n", time_secs_all);
        average_retrieval_time += (time_secs_all/(double)n_queries);
        total_retrieval_time += time_secs_all;

    }


    global_av_distance_similar =  global_av_distance_similar/((float) n_queries);
    global_av_distance_disimilar = global_av_distance_disimilar/((float) n_queries);
    global_min_similar = global_min_similar/((float) n_queries);
    global_min_dissimilar =  global_min_dissimilar/((float) n_queries);
    av_raw_sum = av_raw_sum/((float) n_queries);

    printf("@@@@@@@@@ GLOBAL DISTANCE STATS\n");
    printf("@@@@@@@@@ AV_DIST_SIM=%f, AV_DIST_DISSIM=%f, MIN_DIST_SIM=%f, MIN_DIST_DISSIM=%f\n",global_av_distance_similar, global_av_distance_disimilar, global_min_similar, global_min_dissimilar);
    printf("@@@@@@@@@ AV_RAW_SUM=%f\n", av_raw_sum);

    int t;

    FILE* fp = fopen (filename_results, "a");

    for(t=0;t<9;t++){
            
        printf("AV HITS FOR FIRST %d = %f\n",values_topK[t],av_hits_topK[t]);
        printf("AV PREC FOR FIRST %d = %f\n",values_topK[t],av_prec_topK[t]);
        fprintf(fp, "%d, %f, %f, %f, %f\n", values_topK[t], av_hits_topK[t],av_prec_topK[t],average_retrieval_time,total_retrieval_time);
    }


    fclose(fp);

    printf("Av. Retrieval Time in Seconds: %f \n", average_retrieval_time);

    free(distances_query_dabatase_one_table);
    free(distances_query_dabatase_ensemble);
    free(distances_query_dabatase_minimum);
    free(boolean_groundtruth);

}


void evaluate_hashing_global_knn(hash_ensemble* hashed_database, hash_ensemble* hashed_queries, int n_database, int n_queries, int max_tables, groundtruth_matrix *groundtruth, char* filename_results, int min_k, int pow_k){


    int rad_local_search = 0;

    int n_global_ks = N_GLOBAL_KAS; 
    //The following arrays should be sorted

    int ks_global_search[N_GLOBAL_KAS];

    int idx;
    ks_global_search[0] = min_k;
    for(idx=1;idx<N_GLOBAL_KAS;idx++)
        ks_global_search[idx] = ks_global_search[idx-1]*pow_k;

    //int ks_global_search[N_GLOBAL_KAS] = {25000,30000,35000,40000,45000,50000,60000,70000,80000,100000};
    float precision_list[N_GLOBAL_KAS] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    float recall_list[N_GLOBAL_KAS] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    int hit_counters[N_GLOBAL_KAS] = {0,0,0,0,0,0,0,0,0,0};
    int predicted_counters[N_GLOBAL_KAS] = {0,0,0,0,0,0,0,0,0,0};


    int n_candidates;
    int i,j,k,m,r;
    int *distances_query_dabatase_one_table = calloc(n_database+1, sizeof(int));
    int *distances_query_dabatase_ensemble = calloc(n_database+1, sizeof(int));
    int *distances_query_dabatase_minimum = calloc(n_database+1, sizeof(int));
    int n_original_bits = 20;

    int* boolean_groundtruth = calloc(n_database+1, sizeof(int));
    int SAFE_HAMMING_FOR_SORTING_LAST = max_tables*n_original_bits + 1;
    int BIG_HAMM_FOR_MIN_SEARCH_INITIALIZATION = max_tables*n_original_bits + 1;

    //Range ensemble distances is [0,max_tables*n_bits]
    int range_max = SAFE_HAMMING_FOR_SORTING_LAST;
    int* count_for_sort = calloc(range_max+1, sizeof(int));
    int* sorted_indices_database = calloc(n_database+1, sizeof(int));

    float one_precision;
    float one_recall;

    for(i=0;i<n_queries;i++){
        printf("PROCESSING QUERY %d ...\n",i);
        struct timeval start, end;
        gettimeofday(&start, NULL);

        for(k=0;k<n_database;k++){
            distances_query_dabatase_ensemble[k]=0;
            distances_query_dabatase_minimum[k]=BIG_HAMM_FOR_MIN_SEARCH_INITIALIZATION;
            boolean_groundtruth[k]=0;
        }

        int ndesired=0;
        for(m=0;m<groundtruth->n_cols;m++)
            if(groundtruth->mat[i][m]>=0){
                boolean_groundtruth[groundtruth->mat[i][m]]=1;
                ndesired +=1;
            }

        for(j=0;j<max_tables;j++){
            int n_bits = hashed_queries->tables[j]->n_cols;
            for(k=0;k<n_database;k++){
                distances_query_dabatase_one_table[k]=hamming_distance(hashed_queries->tables[j]->mat[i], hashed_database->tables[j]->mat[k],n_bits);
                
                distances_query_dabatase_ensemble[k]=distances_query_dabatase_ensemble[k]+distances_query_dabatase_one_table[k];
                if(distances_query_dabatase_minimum[k]>distances_query_dabatase_one_table[k])
                    distances_query_dabatase_minimum[k]=distances_query_dabatase_one_table[k];
            }

        }

        //PRE-FILTER WITH LOCAL BALL
        n_candidates = 0;
        for(k=0;k<n_database;k++){
            if(distances_query_dabatase_minimum[k] > rad_local_search){
                distances_query_dabatase_ensemble[k]=SAFE_HAMMING_FOR_SORTING_LAST;
            } else {
                n_candidates++;
            }
        }   

        printf("... %d VALID CANDIDATES\n",n_candidates);

        //GLOBAL KNN SEARCH
        for(m=0;m<=range_max;m++)
            count_for_sort[m]=0;

        for(k=0;k<n_database;k++)
            ++count_for_sort[distances_query_dabatase_ensemble[k]];

        for(m=1;m<=range_max;m++)
            count_for_sort[m]+=count_for_sort[m-1];     

        for(k=0;k<n_database;k++){
            int position = count_for_sort[distances_query_dabatase_ensemble[k]]-1;
            sorted_indices_database[position] = k;
            --count_for_sort[distances_query_dabatase_ensemble[k]]; 
        }

        

        int previous_k=0;
        int previous_hits=0;
        int previous_counters=0;

        for(k=0;k<n_global_ks;k++){
            hit_counters[k]=previous_hits;
            predicted_counters[k]=previous_counters;
            for(r=previous_k;r<ks_global_search[k];r++){//CHECK CORRECTNESS OF PREDICTIONS r=previous_k,previous_k+1 ... current_k
                if(boolean_groundtruth[sorted_indices_database[r]]>0)
                    ++hit_counters[k];
                ++predicted_counters[k];
            }
            previous_k = ks_global_search[k]-1;
            previous_hits = hit_counters[k];
            previous_counters = predicted_counters[k];
        }

        printf("@@@ LOCAL RADIUS %d: %d VALID CANDIDATES FOR QUERY %d\n",rad_local_search,n_candidates, i);

        for(k=0;k<n_global_ks;k++){
            
            if(predicted_counters[k] > 0)
                one_precision=((float)hit_counters[k])/((float)predicted_counters[k]);
            else
                one_precision=1.0;

            one_recall=((float)hit_counters[k])/((float)ndesired);
            precision_list[k]=precision_list[k]+(one_precision/(float)n_queries);
            recall_list[k]=recall_list[k]+(one_recall/(float)n_queries);
            printf("@@@ GLOBAL K %d: HITS %d, PREDICTED %d, DESIRED %d, PRECISION %f, RECALL %f\n",ks_global_search[k],hit_counters[k],predicted_counters[k],ndesired,one_precision,one_recall);
      
        }

        gettimeofday(&end, NULL);
        printf("Time in seconds: %ld \n",
            (((end.tv_sec - start.tv_sec)*1000000L
           +end.tv_usec) - start.tv_usec)/1000000);  
    }

    FILE* fp = fopen (filename_results, "a");

    for(k=0;k<n_global_ks;k++){

       printf("@@@ GLOBAL K %d PRECISION %f, RECALL %f\n",ks_global_search[k],precision_list[k],recall_list[k]);
       fprintf(fp, "%d, %d, %d, %d, %d, %d, %f, %f\n", max_tables, 18, ks_global_search[k], 2, 1, 0, precision_list[k],recall_list[k]);
    }

    fclose(fp);

    free(distances_query_dabatase_one_table);
    free(distances_query_dabatase_ensemble);
    free(distances_query_dabatase_minimum);
    free(count_for_sort);
    free(sorted_indices_database);
    free(boolean_groundtruth);

}

int main(int argc, char *argv[])
{
    struct timeval start, end;
    gettimeofday(&start, NULL);
    char* groundtruth_filename = argv[1];
    char* results_filename = argv[2];

    //char filename[] = "Results_BOHASHER_NUSWIDE.txt";

    check_errors(argc,argv);

    int aggregation_mode = atoi(argv[3]); //aggregation_mode==0 -> KNN
    int max_models = atoi(argv[4]);

    int min_k = 100;
    int pow_k = 2;
    float min_r = 1.0;
    float max_r = 5.0; 

    int nbits = atoi(argv[5]);

    hash_ensemble* hashed_database = malloc(sizeof(hash_ensemble));
    hashed_database->n_tables = max_models;
    hashed_database->tables = calloc(hashed_database->n_tables+1,sizeof(matrix*));

    ///READ WEIGHTS
    char weights_name[100];
    sprintf(weights_name, "BoHash_CIFAR_WEIGHTS.txt");
    float* table_weights = calloc(max_models+1,sizeof(float));
    FILE* ffile;
    ffile=fopen(weights_name,"r");
    if (ffile==NULL)
    {
        char message[23]="Error abriendo archivo ";
        error(strcat(message,weights_name));
    }
    int i;
    float weight;
    for(i=0; i < max_models; i++){
        fscanf(ffile,"%f ",&weight);
        table_weights[i] = weight;
    }
    ///END READ WEIGHTS

    printf("Inicializando tablas database ...\n");

    int t;
    for(t=0;t<hashed_database->n_tables;t++){
        char table_file_name[100];
        sprintf(table_file_name, "BoHash_CIFAR_Database_%dBits_Model%d.txt", nbits, t);
        hashed_database->tables[t] = malloc(sizeof(matrix));
        hashed_database->tables[t] = fill_matrix(hashed_database->tables[t],table_file_name);
        printf("... table %d done.\n",t);
    }
    
    int n_database = hashed_database->tables[0]->n_rows;


    hash_ensemble* hashed_queries = malloc(sizeof(hash_ensemble));
    hashed_queries->n_tables = max_models;
    hashed_queries->tables = calloc(hashed_queries->n_tables+1,sizeof(matrix*));

    printf("Inicializando tablas queries\n");

    for(t=0;t<hashed_queries->n_tables;t++){
        char table_file_name[100];
        sprintf(table_file_name, "BoHash_CIFAR_Queries_%dBits_Model%d.txt", nbits, t);
        hashed_queries->tables[t] = malloc(sizeof(matrix));
        hashed_queries->tables[t] = fill_matrix(hashed_queries->tables[t],table_file_name);
        printf("... table %d done.\n",t);
    }

    int n_queries = hashed_queries->tables[0]->n_rows;


    int n_groundtruth_neighbors = 5700;
    groundtruth_matrix *groundtruth = malloc(sizeof(groundtruth_matrix));
    groundtruth = fill_groundtruth_matrix(groundtruth,groundtruth_filename,n_queries,n_groundtruth_neighbors);

    if(aggregation_mode == 0){

        evaluate_hashing_global_knn(hashed_database, hashed_queries, n_database, n_queries, max_models, groundtruth, results_filename, min_k, pow_k);

    } else {

        evaluate_hashing_global_ball(hashed_database, hashed_queries, n_database, n_queries, max_models, groundtruth, results_filename, min_r, max_r, table_weights);

    }
    
    //querys = fill_matrix(querys,query_file);
    //recover1 = fill_matrix(recover1,recover_file);
   
    /*
    printf("Computing Distances ...\n");
    distance_matrix* hamm_matrix = malloc(sizeof(distance_matrix));
    distance_matrix* hamm_matrix2 = malloc(sizeof(distance_matrix));
    hamm_matrix = all_pairwise_distances(hamm_matrix, querys, recover);
    hamm_matrix2 = all_pairwise_distances(hamm_matrix2, querys, recover2);
    printf("DONE\n");
    
    printf("Liberar espacio\n");
    free_distance_matrix(hamm_matrix);
    */

    free_ensemble(hashed_queries);
    free_ensemble(hashed_database);

    gettimeofday(&end, NULL);
    printf("Time in seconds: %ld \n",
            (((end.tv_sec - start.tv_sec)*1000000L
           +end.tv_usec) - start.tv_usec)/1000000);
    return 0;
}
