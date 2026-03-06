import java.util.ArrayList;
import java.util.Scanner;
import java.util.HashMap;
import java.util.HashSet;

public class DiffDataStructures {
    // searching - linear search
      public static int linearSearch(int[] arr,int target){
          for(int i=0;i<arr.length;i++){
              return i;
          }
          return -1;
      }
      //Binary Search
      public static int BinarySearch(int[] arr,int target){
          int low=0,high = arr.length-1;
          while(low<= high){
              int mid =(low+high)/2;
              if(arr[mid]== target){
                  return mid;
              }else if(arr[mid]<target){
                  low = mid+1;
              }else{
                  high = mid-1;
              }
          }
          return -1;
      }
      public static void main(String[] args){
          // fix sized array
          int[] arr = new int[5];
          int[] arr1 ={10,20,30,40};
          for(int i=0;i<arr1.length;i++){
              System.out.println(arr1[i]);
          }
          //insertion at end
          int n = 4;
          arr[n]=22;
          // INsertion at index , shift elements
          int index = 1;
          int value = 11;
          for(int i=n-1;i>index;i--){
              arr[i] = arr[i-1];
          }
          arr[index]=value;


          //Deletion at index (left sift)
          for(int i=index;i<n-1;i++){
              arr[i]=arr[i+1];
          }



      }

}
