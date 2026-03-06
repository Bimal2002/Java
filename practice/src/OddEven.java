import java.util.Scanner;

public class OddEven {
   public static void main(String[] args){
       Scanner scanner = new Scanner(System.in);
       System.out.printf("Enter the number : ");
       int num = scanner.nextInt();
       if(num %2 !=0){
           System.out.printf("Odd number");
       }else{
           System.out.printf("Even number");
       }
   }
}
