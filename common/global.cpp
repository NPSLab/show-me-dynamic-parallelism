#include "global.h"

list<int> *adjacencyNodeList;
list<int> *adjacencyWeightList;

struct _GRAPH_ graph;
char buff[BUFF_SIZE] = {};
int noNodeTotal = 0;
int noEdgeTotal = 0;
int source = 0;
FILE* fp = NULL;
int VERBOSE;
int DEBUG;

float *bc;

double gettime() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec+t.tv_usec*1e-6;
}

double gettime_ms() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec*1e+3+t.tv_usec*1e-3;
}

int readInputDIMACS9()
{
    char *p;
    int tailNode, headNode, edgeWeight;
    while( fgets(buff, BUFF_SIZE, fp) != NULL ){
        if ( buff[0] == 'c' )   continue;
        if ( buff[0] == 'p' ){
            p = strtok(buff+4, " ");
            noNodeTotal = atoi(p);
            p = strtok( NULL, " ");
            noEdgeTotal = atoi(p);
            adjacencyNodeList = new list<int> [noNodeTotal];
        	adjacencyWeightList = new list<int> [noNodeTotal];
		}
        if ( buff[0] == 's' ){
            p = strtok(buff+1, " ");
            source = atoi(p) - 1;
        }
        if ( buff[0] == 'a' ){
            /* get tail vertex no */
            p = strtok(buff+1, " ");
            tailNode = atoi(p) - 1;
            /* get head vertex no */
            p = strtok(NULL, " ");
            headNode = atoi(p) - 1;
            /* get edge value no */
            p = strtok(NULL, " ");
            edgeWeight = atoi(p);
            //printf("%d %d %d\n", tailNode, headNode,edgeWeight);
            adjacencyNodeList[tailNode].push_back(headNode);
			adjacencyWeightList[tailNode].push_back(edgeWeight);
        }
    }
	
	return 0;
}

int readInputDIMACS10()
{
    //int degree = 0;
	char *p;
    int tailNode, headNode, edgeWeight;
    while( fgets(buff, BUFF_SIZE, fp) != NULL ){ // read comments
        if ( buff[0] == '%' )   continue;
		else	break;
	}
	//printf("%d\n", strlen(buff));
	p = strtok(buff," ");
	noNodeTotal = atoi(p);
	p = strtok(NULL, " ");
	noEdgeTotal = atoi(p);
	noEdgeTotal = 2 * noEdgeTotal;

	adjacencyNodeList = new list<int> [noNodeTotal];
	adjacencyWeightList = new list<int> [noNodeTotal];
	srand( 7 );
	for (int i=0; i<noNodeTotal; ++i){
		tailNode = i;
		if ( fgets(buff, BUFF_SIZE, fp)==NULL ) {
			EXIT("Parsing adjacencylist fails!");
		}
		if ( strlen(buff) == BUFF_SIZE-1 )
			printf("line: %d\n", i);
		assert ( strlen(buff) != BUFF_SIZE-1 );
		p = strtok(buff, " ");
		while( p!=NULL && (*p)!='\n' ){
			//degree++;
			headNode = atoi(p) - 1;
			adjacencyNodeList[tailNode].push_back(headNode);
			int weight = rand() % 1000 + 1;			
			adjacencyWeightList[tailNode].push_back(weight);
			p = strtok(NULL, " ");
		}
	}

    return 0;
}

int readInputSLNDC()
{
	/* input raw data */
    char *p;
    int tailNode, headNode, edgeWeight = 1;
	int maxNodeNo = 0;
	list<int> *tempNodeList;
	int *mapTable;
	if ( fgets(buff, 256, fp)==NULL ) {
		EXIT("");
	}
	if ( fgets(buff, 256, fp)==NULL ) {
		EXIT("");
	}
	/* parse node number and edge number */
	if ( fgets(buff, 256, fp)==NULL ) {
		EXIT("Parsing total node number fails!");
	}
	p = strtok(buff+9, " ");
    noNodeTotal = atoi(p);
    p = strtok( NULL, " ");		p = strtok( NULL, " "); 
	noEdgeTotal = atoi(p);
	if ( fgets(buff, 256, fp)==NULL ) {
		EXIT("");
	}
	
	tempNodeList = new list<int> [noNodeTotal*2];
	mapTable = new int [noNodeTotal*2] ();
	noEdgeTotal = 0;
	while( fscanf(fp, "%d%d", &tailNode, &headNode)!= EOF ){
        //printf("%d %d %d\n", tailNode, headNode,edgeWeight);
        //printf("%d %d\n", tailNode, headNode);
		if ( tailNode>=noNodeTotal*2 || headNode>=noNodeTotal*2 ) {
			printf("Tail node: %d\nHead node: %d\n", tailNode, headNode);
			exit(0);
		}
		mapTable[tailNode] = mapTable[headNode] = 1;
		if ( tailNode>maxNodeNo )
			maxNodeNo = tailNode;
		if ( headNode>maxNodeNo )
			maxNodeNo = headNode;
        tempNodeList[tailNode].push_back(headNode);
		noEdgeTotal++;
	}

	/* build mapping table */
	noNodeTotal = 0;
	for (int i=0; i<=maxNodeNo; ++i){
		if ( mapTable[i]==1 ){
			mapTable[i] = noNodeTotal;
			noNodeTotal++;
		}
		else
			mapTable[i] = -1;
	}

	/* eliminate discrete node */
	adjacencyNodeList = new list<int> [noNodeTotal];
	adjacencyWeightList = new list<int> [noNodeTotal];
	noNodeTotal = 0;
	srand( 7 );
	for (int i=0; i<=maxNodeNo; ++i){
		if ( mapTable[i] == -1 )	// no mapping
			continue;
		else{						// convert to contiguous adjacencylist
			tailNode = mapTable[i];
			while ( !tempNodeList[i].empty() ){
				int tmpHeadNode = tempNodeList[i].front();
				tempNodeList[i].pop_front();
				headNode = mapTable[tmpHeadNode];
				if (headNode==-1)
					printf("Error for mapTable, %d\n", tmpHeadNode);
				adjacencyNodeList[tailNode].push_back(headNode);
				int weight = rand() % 1000 + 1;
				adjacencyWeightList[tailNode].push_back(weight);
			}
			noNodeTotal++;
			//printf("Size of node %d is %d\n", tailNode, adjacencyNodeList[tailNode].size() );
		}
	}
	delete [] tempNodeList;
	delete [] mapTable;
    return 0;
}

int convertCSR()
{
	int startingPos, noEdgePerNode;
	graph.vertexArray = new int [noNodeTotal+1] ();
	graph.costArray = new int [noNodeTotal] ();
	graph.levelArray = new int [noNodeTotal] ();
	graph.edgeArray = new int [noEdgeTotal] ();
	graph.weightArray = new int [noEdgeTotal] ();
	graph.frontier = new char [noNodeTotal] ();
	graph.update = new char [noNodeTotal] ();
	graph.visited = new char [noNodeTotal] ();
	memset(graph.costArray, 0, sizeof(int)*noNodeTotal);
	memset(graph.levelArray, 0, sizeof(int)*noNodeTotal);
	memset(graph.frontier, 0, sizeof(char)*noNodeTotal);
	memset(graph.update, 0, sizeof(char)*noNodeTotal);
	memset(graph.visited, 0, sizeof(char)*noNodeTotal);

	startingPos = 0;
	noEdgePerNode = 0;
	for (int i=0; i<noNodeTotal; ++i ){
		startingPos = startingPos + noEdgePerNode;
		noEdgePerNode = 0;
		graph.vertexArray[i] = startingPos;
		//printf("Node %d is connected to :", i+1);
		while ( adjacencyNodeList[i].empty()!=true && adjacencyWeightList[i].empty()!=true ){
			graph.edgeArray[ startingPos + noEdgePerNode ] = adjacencyNodeList[i].back();
			graph.weightArray[ startingPos + noEdgePerNode ] =  adjacencyWeightList[i].back();
			adjacencyNodeList[i].pop_back();
			adjacencyWeightList[i].pop_back();
			//printf("%d(%d)\t",graph.edgeArray[startingPos+noEdgePerNode]+1, graph.weightArray[startingPos+noEdgePerNode]);
			noEdgePerNode++;	
		}
		//printf("\n");
	}
	graph.vertexArray[noNodeTotal] = noEdgeTotal;
	if ( VERBOSE ) {
		fprintf(stderr, "Node no: %d\n", noNodeTotal);
		fprintf(stderr, "Edge no: %d\n", noEdgeTotal);
	}
	delete [] adjacencyNodeList;
	delete [] adjacencyWeightList;
	return 0;

}

int invertGraph()
{
	list<int> *childAdjNodeList = new list<int> [noNodeTotal];
	if ( graph.childVertexArray!=NULL )
		delete [] graph.childVertexArray;
	if ( graph.rEdgeArray!=NULL )
		delete [] graph.rEdgeArray;

	graph.childVertexArray = new int [noNodeTotal+1] ();
	graph.rEdgeArray = new int [noEdgeTotal] ();
	int index = 0;
	for (int i=0; i<noNodeTotal; ++i) {
		int start = graph.vertexArray[i];
		int end = graph.vertexArray[i+1];
		for (int j=start; j<end; ++j) {
			int nid = graph.edgeArray[j];
			childAdjNodeList[nid].push_back(i);
			++index;
		}
	}
	//printf("Reversed edges : %d\n", index);

	int startingPos = 0;
	int noEdgePerNode = 0;
	for (int i=0; i<noNodeTotal; ++i) {
		startingPos = startingPos + noEdgePerNode;
		noEdgePerNode = 0;
		graph.childVertexArray[i] = startingPos;
		while ( childAdjNodeList[i].empty()!=true ) {
			graph.rEdgeArray[ startingPos + noEdgePerNode ] = childAdjNodeList[i].back();
			childAdjNodeList[i].pop_back();
			++noEdgePerNode;
		}
		//if ( i%100000==0 )
		//  printf("Finish %d nodes...\n", i);
	}
	graph.childVertexArray[noNodeTotal] = noEdgeTotal;
	delete [] childAdjNodeList;
 
	return 0;
}

// Multiple DFS
void BFS_queue_init(queue<int> &myqueue, int depth, int degree)
{
	// initialize all level
	for (int i=0; i<noNodeTotal; ++i){
		graph.levelArray[i] = MAX_LEVEL;
	}
	myqueue.push(source);
	graph.visited[source] = 1;
	graph.frontier[source] = 1;
	graph.levelArray[source] = 0;
	int start = graph.vertexArray[source];
	int end = graph.vertexArray[source+1];		
	if ( end-start>degree )
		end = start + degree;
	for (int i=start; i<end; ++i){	// DFS 
		int curr = graph.edgeArray[i];
		//printf("From %d(%d)", source, graph.levelArray[source]);
		myqueue.push(curr);
		graph.frontier[curr] = 1;
		for (int j=1; j<=depth; j++){
			int level = j;
			graph.levelArray[curr] = level;
			//printf(" -> %d(%d)", curr, graph.levelArray[curr]);			
			int nextStart = graph.vertexArray[curr];
			int nextEnd = graph.vertexArray[curr+1];
			int k;		
			for (k=nextStart; k<nextEnd; ++k) {
				curr = graph.edgeArray[k];
				if ( graph.levelArray[curr]==MAX_LEVEL ){	// unvisited node, break out the for loop
					myqueue.push(curr);
					graph.frontier[curr] = 1;	
					break;
				}
			}
			if (k==nextEnd)	break;
		}
		//printf("\n");
	}
}

// Multiple DFS
void SSSP_queue_init(queue<int> &myqueue, int depth, int degree)
{
	// initialize all level
	for (int i=0; i<noNodeTotal; ++i){
		graph.costArray[i] = INF;
	}
	myqueue.push(source);
	graph.frontier[source] = 1;
	graph.costArray[source] = 0;
	int start = graph.vertexArray[source];
	int end = graph.vertexArray[source+1];		
	if ( end-start>degree )
		end = start + degree;
	for (int i=start; i<end; ++i){	// DFS 
		int curr = graph.edgeArray[i];
		//printf("From %d(%d)", source, graph.levelArray[source]);
		myqueue.push(curr);
		graph.frontier[curr] = 1;
		int pre = graph.costArray[curr] = graph.weightArray[i];
		for (int j=1; j<=depth; j++){
			//printf(" -> %d(%d)", curr, graph.levelArray[curr]);			
			int nextStart = graph.vertexArray[curr];
			int nextEnd = graph.vertexArray[curr+1];
			int k;
			for (k=nextStart; k<nextEnd; ++k) {
				curr = graph.edgeArray[k];
				if ( graph.levelArray[curr]==INF ){	// unvisited node, break out the for loop
					myqueue.push(curr);
					graph.frontier[curr] = 1;
					graph.costArray[curr] = graph.weightArray[k]+pre;
					pre = graph.costArray[curr];
					break;
				}
			}
			if (k==nextEnd)	break;
		}
		//printf("\n");
	}
}

int outputLevel()
{
	for (int i=0; i<noNodeTotal; ++i) {
		printf("Node %d is in level %d\n", i+1, graph.levelArray[i]);
	}
	return 0;
}

int outputCost(FILE *file=stdout)
{
	for (int i=0; i<noNodeTotal; ++i) {
		fprintf(file,"From Node %d -> %d is %d\n", source+1, i+1, graph.costArray[i]);
	}
	return 0;
}

int outputBC(FILE *file=stdout)
{
	fprintf(stderr,"------- BC value ------\n");
	for (int i=0; i<noNodeTotal; ++i) {
		fprintf(file,"Node %d: %.2f\n", i+1, bc[i]);
	}
}

int outputRank(FILE *file=stdout)
{
	fprintf(stderr,"------- Rank value ------\n");
	for (int i=0; i<noNodeTotal; ++i) {
		fprintf(file,"Node %d: %0.6lf\n", i+1, graph.rankArray[i]);
	}
}

int outputColor(FILE *file=stdout)
{
	fprintf(stderr,"------- Color value ------\n");
	for (int i=0; i<noNodeTotal; ++i) {
		fprintf(file,"Node %d: %d\n", i+1, graph.colorArray[i]);
	}
}

int clear()
{
	delete [] graph.vertexArray;
	delete [] graph.edgeArray;
	delete [] graph.weightArray;
	delete [] graph.frontier;

	return 0;
}
