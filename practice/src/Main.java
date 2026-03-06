
import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.printf("Enter first value : ");
        int x = scanner.nextInt();
        System.out.printf("Enter second value : ");
        int y = scanner.nextInt();
        System.out.printf("Enter Operation sign : ");
        char op = scanner.next().charAt(0);
        if(op == '+'){
            System.out.printf("Final Value : " + (x+y));
        }else if(op == '-'){
            System.out.printf("Final Value : " + (x-y));
        }else if(op == '*'){
            System.out.printf("Final Value : "+ (x*y));
        }else{ // div
            if(y !=0){
                System.out.printf("Final Value : "+ (x/y));
            }else{
                System.out.printf("Error not divisible ");
            }

        }
    }
}